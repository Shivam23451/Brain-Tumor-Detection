const fileInput = document.getElementById("mri_image");
const fileName = document.getElementById("file-name");
const preview = document.getElementById("image-preview");

fileInput.addEventListener("change", function () {

    const file = fileInput.files[0];

    if (!file) {
        fileName.textContent = "No image selected";
        preview.style.display = "none";
        return;
    }

    fileName.textContent = "Selected: " + file.name;

    const imageURL = URL.createObjectURL(file);

    preview.src = imageURL;
    preview.style.display = "block";
});