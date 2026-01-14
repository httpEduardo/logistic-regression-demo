const output = document.getElementById("output") as HTMLPreElement;
const trainButton = document.getElementById("trainButton") as HTMLButtonElement;
const predictButton = document.getElementById("predictButton") as HTMLButtonElement;

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

trainButton.addEventListener("click", () => {
  const rows = (document.getElementById("rowsInput") as HTMLTextAreaElement).value;
  const epochs = parseInt((document.getElementById("epochsInput") as HTMLInputElement).value, 10);
  const lr = parseFloat((document.getElementById("lrInput") as HTMLInputElement).value);
  fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows, label: "label", epochs, lr }),
  })
    .then((res) => res.json())
    .then((data) => {
      output.textContent = pretty(data);
    });
});

predictButton.addEventListener("click", () => {
  const row = (document.getElementById("rowInput") as HTMLInputElement).value;
  fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row }),
  })
    .then((res) => res.json())
    .then((data) => {
      output.textContent = pretty(data);
    });
});
