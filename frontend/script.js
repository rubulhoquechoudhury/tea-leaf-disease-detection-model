document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result');
    const submitButton = uploadForm.querySelector('button[type="submit"]');

    // Add loading state management
    function setLoading(isLoading) {
        submitButton.disabled = isLoading;
        if (isLoading) {
            resultDiv.innerHTML = '<div class="loading active">Processing image...</div>';
        }
    }

    // Add file validation
    function validateFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
        const maxSize = 5 * 1024 * 1024; // 5MB

        if (!validTypes.includes(file.type)) {
            throw new Error('Please upload a valid image file (JPEG, PNG, or GIF)');
        }

        if (file.size > maxSize) {
            throw new Error('File size should be less than 5MB');
        }
    }

    // Handle form submission
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent form from submitting traditionally
        
        const file = imageInput.files[0];

        try {
            if (!file) {
                throw new Error('Please select an image.');
            }

            validateFile(file);
            setLoading(true);

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to process image');
            }

            const data = await response.json();
            resultDiv.className = 'success';
            resultDiv.innerHTML = `
                <div class="prediction-result">
                    <h3>Prediction Result:</h3>
                    <p>${data.prediction}</p>
                </div>
            `;
        } catch (error) {
            resultDiv.className = 'error';
            resultDiv.innerHTML = `
                <div class="error-message">
                    <h3>Error:</h3>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            setLoading(false);
        }
    });

    // Add file input change handler
    imageInput.addEventListener('change', () => {
        const file = imageInput.files[0];
        if (file) {
            try {
                validateFile(file);
                resultDiv.textContent = '';
                resultDiv.className = '';
            } catch (error) {
                resultDiv.className = 'error';
                resultDiv.innerHTML = `
                    <div class="error-message">
                        <h3>Error:</h3>
                        <p>${error.message}</p>
                    </div>
                `;
                imageInput.value = '';
            }
        }
    });
}); 