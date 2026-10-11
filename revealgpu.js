// revealgpu: run the WGSL kernel shown on a reveal.js slide on the viewer's own GPU (WebGPU).
//
//   <div class="revealgpu" data-n="1048576" data-input="i" data-expect="x === 2 * i">
//     <pre><code class="language-rust"><script type="text/template">
//   @group(0) @binding(0) var<storage, read_write> data : array<u32>;
//   @compute @workgroup_size(64)
//   fn main(@builtin(global_invocation_id) id : vec3<u32>) {
//     if (id.x < arrayLength(&data)) { data[id.x] = 2u * data[id.x]; }
//   }
//     </script></code></pre>
//   </div>
//
// The code goes on the left, a Run button and its output on the right. Run uploads n u32 values
// (data-input, a JS expression in i), runs the kernel once over them, reads them back and checks
// every value (data-expect, a JS expression in x, i and input), then times data-runs more runs.

const CSS = `
.reveal .revealgpu { display: grid; grid-template-columns: minmax(0, 1fr) var(--revealgpu-panel, 300px); gap: 24px;
  align-items: start; text-align: left; }
.reveal .revealgpu pre { width: 100%; margin: 0; font-size: 18px; }
.reveal .revealgpu button { font: 600 22px/1.2 sans-serif; padding: 10px 20px; border: 0; background: #000; color: #fff;
  cursor: pointer; }
.reveal .revealgpu button:disabled { opacity: .5; cursor: wait; }
.reveal .revealgpu output { display: block; margin-top: 14px; padding: 14px; min-height: 120px; background: #f4f4f4;
  font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }
.reveal .revealgpu .ok { color: #1a7f37; } .reveal .revealgpu .bad { color: #c62828; }`;

let device;   // one device for the whole deck
async function getDevice() {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  device ??= navigator.gpu.requestAdapter().then(async (adapter) => {
    if (!adapter) throw new Error('no GPU adapter found');
    const d = await adapter.requestDevice();
    const i = adapter.info || {};
    d.label = [i.vendor, i.architecture].filter(Boolean).join(', ') || 'a GPU';
    d.lost.then(() => { device = undefined; });
    return d;
  });
  return device;
}

async function run(el, code, log) {
  const n = Number(el.dataset.n || 1024), runs = Math.max(1, Number(el.dataset.runs || 5));
  const fill = new Function('i', `return (${el.dataset.input || 'i'});`);
  const input = Uint32Array.from({ length: n }, (_, i) => fill(i));
  const gpu = await getDevice();
  log(`GPU: ${gpu.label}`);

  const module = gpu.createShaderModule({ code });
  const errors = (await module.getCompilationInfo()).messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(errors.map((m) => `line ${m.lineNum}: ${m.message}`).join('\n'));
  const size = Number(/@workgroup_size\s*\(\s*(\d+)/.exec(code)?.[1] || 64);
  const groups = Math.ceil(n / size);
  if (groups > gpu.limits.maxComputeWorkgroupsPerDimension) throw new Error(`n is too large for one dispatch of workgroups of ${size}`);

  gpu.pushErrorScope('validation');
  const pipeline = gpu.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: el.dataset.entry } });
  const data = gpu.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const back = gpu.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bind = gpu.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: data } }] });
  const once = async (copyBack) => {
    const enc = gpu.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    if (copyBack) enc.copyBufferToBuffer(data, 0, back, 0, n * 4);
    const t0 = performance.now();
    gpu.queue.submit([enc.finish()]);
    await gpu.queue.onSubmittedWorkDone();
    return performance.now() - t0;
  };
  try {
    gpu.queue.writeBuffer(data, 0, input);
    await once(true);
    const error = await gpu.popErrorScope();
    if (error) throw new Error(error.message);
    await back.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(back.getMappedRange().slice(0));
    back.unmap();

    if (el.dataset.expect) {
      const ok = new Function('x', 'i', 'input', `return (${el.dataset.expect});`);
      const bad = out.findIndex((x, i) => !ok(x, i, input));
      if (bad < 0) log(`✓ all ${n.toLocaleString('en-US')} values match`, 'ok');
      else log(`✗ data[${bad}] is ${out[bad]}`, 'bad');
    }
    const times = [];
    for (let r = 0; r < runs; r++) times.push(await once(false));
    times.sort((a, b) => a - b);
    log(`time: ${times[runs >> 1].toFixed(2)} ms (median of ${runs} runs)`);
    log(`data: ${Array.from(out.slice(0, 8)).join(' ')}${n > 8 ? ' …' : ''}`);
  } finally {
    data.destroy();
    back.destroy();
  }
}

function setup(el) {
  const code = el.querySelector('code')?.textContent.trim() || '';
  const panel = document.createElement('div');
  panel.innerHTML = `<button>${el.dataset.label || 'Run on this GPU'}</button><output>press Run</output>`;
  el.append(panel);
  const [button, output] = panel.children;
  const log = (text, cls) => {
    const line = document.createElement('div');
    line.textContent = text;
    if (cls) line.className = cls;
    output.append(line);
  };
  button.addEventListener('click', async () => {
    button.disabled = true;
    output.textContent = '';
    try { await run(el, code, log); } catch (e) { log(`✗ ${e.message}`, 'bad'); }
    button.disabled = false;
  });
}

export default () => ({
  id: 'revealgpu',
  init(deck) {
    document.head.insertAdjacentHTML('beforeend', `<style>${CSS}</style>`);
    deck.getRevealElement().querySelectorAll('.revealgpu').forEach(setup);
  },
});
