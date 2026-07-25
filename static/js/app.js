const menuBtn = document.getElementById("menuBtn");
const navLinks = document.getElementById("navLinks");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const previewImage = document.getElementById("previewImage");

if (menuBtn && navLinks) {
  menuBtn.addEventListener("click", () => navLinks.classList.toggle("open"));
}

if (fileInput && fileName && previewImage) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) {
      fileName.textContent = "Choose a chest X-ray image";
      previewImage.style.display = "none";
      return;
    }
    fileName.textContent = file.name;
    const reader = new FileReader();
    reader.onload = (event) => {
      previewImage.src = event.target.result;
      previewImage.style.display = "block";
    };
    reader.readAsDataURL(file);
  });
}
