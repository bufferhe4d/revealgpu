# revealgpu

A reveal.js plugin that runs the WGSL kernel shown on a slide on the viewer's own GPU, in the
browser, with WebGPU. One file, no dependencies.

The block puts the kernel on the left and a **Run on this GPU** button on the right. Run uploads
the input, runs the kernel, checks every result and times it:

```
GPU: apple, metal-3
✓ all 1,048,576 values match
time: 0.40 ms (median of 5 runs)
data: 5219 5085 3046 7668 6688 9970 3036 1533 …
```

Try `index.html`: serve this folder (`python3 -m http.server`) and open http://localhost:8000.

## Use

```html
<script type="module">
  import Reveal from 'https://cdn.jsdelivr.net/npm/reveal.js@6/dist/reveal.mjs';
  import RevealGPU from './revealgpu.js';
  Reveal.initialize({ plugins: [RevealGPU] });
</script>
```

```html
<section>
  <div class="revealgpu" data-n="1048576" data-input="i" data-expect="x === 2 * i">
    <pre><code class="language-rust" data-trim><script type="text/template">
@group(0) @binding(0) var<storage, read_write> data : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < arrayLength(&data)) { data[id.x] = 2u * data[id.x]; }
}
    </script></code></pre>
  </div>
</section>
```

| attribute | default | meaning |
|---|---|---|
| `data-n` | `1024` | number of `u32` values in `data` |
| `data-input` | `i` | JavaScript expression in `i`: the starting value of `data[i]` |
| `data-expect` | none | JavaScript expression in `x`, `i`, `input`: true when `x`, the result in `data[i]`, is right. Every value is checked. |
| `data-runs` | `5` | timed runs after the checked one; the median is shown |
| `data-entry` | the only one | compute entry point, if the module has several |
| `data-label` | `Run on this GPU` | button text |

## The kernel

- Group 0, binding 0 is `var<storage, read_write> data : array<u32>`, holding `data-n` values.
- One invocation per value: Run dispatches `ceil(n / size)` workgroups, where `size` is the first
  number in `@workgroup_size(…)`. Guard with `arrayLength(&data)`, since the last workgroup can run
  past the end.
- One dispatch allows at most 65,535 workgroups, so `n` is at most 65,535 × `size`.

Run does `writeBuffer` (the input), one dispatch, a copy back and the check, then `data-runs` more
dispatches, each timed from submit until the GPU reports the work done.

## Notes

- WebGPU only works on `https://` pages and on `localhost`.
- highlight.js has no WGSL grammar; `language-rust` is close. The `<script type="text/template">`
  wrapper (a reveal.js feature) saves escaping `<` and `>`.
- The panel is 300px wide; set `--revealgpu-panel` to change it. Styles are injected, so override
  them with `.reveal .revealgpu …`.
- `data-input` and `data-expect` run as JavaScript. Use them on your own slides only.

## License

MIT
