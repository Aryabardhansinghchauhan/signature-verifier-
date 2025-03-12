document.getElementById("upload-button").addEventListener("click", async () => {
    const fileInput = document.getElementById("file-input");
    const resultDiv = document.getElementById("result");
    const loader = document.getElementById("loader");
    const consentCheckbox = document.getElementById("consent-checkbox");
    const errorMessage = document.getElementById("error-message");

    errorMessage.style.display = "none";
    errorMessage.textContent = "";

    if (fileInput.files.length === 0) {
        errorMessage.textContent = "No file chosen. Please upload an image file.";
        errorMessage.style.display = "block"; 
        return; 
    }
    
    if (!consentCheckbox.checked) {
        errorMessage.textContent = "You must consent to the processing of the image to proceed.";
        errorMessage.style.display = "block"; 
        return; 
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    loader.style.display = "block";
    resultDiv.innerHTML = "";

    try {
        const response = await fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        });

        loader.style.display = "none";

        if (response.ok) {
            const result = await response.json();
            resultDiv.innerHTML = `<p>Result: ${result.label}</p>`;
        } else {
            const error = await response.json();
            errorMessage.textContent = error.error;
            errorMessage.style.display = "block"; 
        }
    } catch (error) {
        loader.style.display = "none";
        errorMessage.textContent = "An error occurred while processing the image.";
        errorMessage.style.display = "block"; 
    }
});


