const fileSelect = document.getElementById("fileSelect"),
    fileElem = document.getElementById("fileElem"),
    fileList = document.getElementById("fileList"),
    btn = document.getElementById("upload"),
    result = document.getElementById("result"),
    selectLink = document.getElementById("link");

btn.addEventListener("click", uploadFile);
selectLink.addEventListener(
    "click",
    (e) => {
        if (fileElem) {
            fileElem.click();
        }
        e.preventDefault();
    },
    false,
);
fileSelect.addEventListener(
    "click",
    (e) => {
        if (fileElem) {
            fileElem.click();
        }
        e.preventDefault();
    },
    false,
);

fileElem.addEventListener("change", handleFiles, false);

function handleFiles() {
    if (!this.files.length) {
        btn.disabled = "disabled"
        fileList.innerHTML = "<p>没有选择任何文件！</p>";
    } else {
        btn.disabled = ""

        fileList.innerHTML = "";
        const img = document.createElement("img");
        img.src = URL.createObjectURL(this.files[0]);
        img.height = 200;
        img.className += "card-img-top"
        img.onload = () => {
            URL.revokeObjectURL(img.src);
        };
        fileList.appendChild(img);
        const info = document.createElement("span");
        info.innerHTML = `${this.files[0].name}`;
        fileList.appendChild(info);
    }
}

function uploadFile() {
    const file = fileElem.files[0];
    const formData = new FormData();

    formData.append('image', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            result.innerHTML = data.message;
        })
        .catch(error => {
            console.error('Error:', error);
        });
}