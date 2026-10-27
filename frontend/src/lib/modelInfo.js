// The notebook's Conv1D network, with output shapes and parameter counts worked out.
export function modelLayers(nFeatures = 162, nClasses = 8) {
  const layers = [];
  let len = nFeatures, ch = 1;
  const conv = (filters) => {
    layers.push({ name: `Conv1D (${filters} filters, kernel 5, ReLU)`, shape: [len, filters], params: 5 * ch * filters + filters });
    ch = filters;
  };
  const pool = () => {
    len = Math.ceil(len / 2);
    layers.push({ name: "MaxPooling1D (pool 5, stride 2)", shape: [len, ch], params: 0 });
  };
  layers.push({ name: "Input (162 features)", shape: [len, 1], params: 0 });
  conv(256); pool();
  conv(256); pool();
  conv(128); pool();
  layers.push({ name: "Dropout (0.2)", shape: [len, ch], params: 0 });
  conv(64); pool();
  const flat = len * ch;
  layers.push({ name: "Flatten", shape: [flat], params: 0 });
  layers.push({ name: "Dense (32, ReLU)", shape: [32], params: flat * 32 + 32 });
  layers.push({ name: "Dropout (0.3)", shape: [32], params: 0 });
  layers.push({ name: `Dense (${nClasses}, softmax)`, shape: [nClasses], params: 32 * nClasses + nClasses });
  return { layers, total: layers.reduce((s, l) => s + l.params, 0) };
}
