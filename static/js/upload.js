// Drag and drop functionality
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// Drag and drop events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
        showAlert('Please select an image file (JPG, PNG, JPEG)', 'danger');
        return;
    }

    // Preview image
    const reader = new FileReader();
    reader.onload = function(e) {
        preview.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    fileInput.value = '';
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    results.style.display = 'none';
}

async function analyzeImage() {
    const file = fileInput.files[0];
    if (!file) {
        showAlert('Please select an image first', 'warning');
        return;
    }

    analyzeBtn.disabled = true;
    loading.style.display = 'block';
    results.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            showAlert(data.error, 'danger');
        } else {
            displayResults(data);
        }
    } catch (error) {
        showAlert('Error analyzing image: ' + error.message, 'danger');
    } finally {
        analyzeBtn.disabled = false;
        loading.style.display = 'none';
    }
}

function displayResults(data) {
    results.style.display = 'block';
    results.innerHTML = `
        <div class="alert alert-${data.prediction === 'NORMAL' ? 'success' : 'danger'}">
            <h4><i class="fas fa-${data.prediction === 'NORMAL' ? 'check' : 'exclamation-triangle'} me-2"></i>
                Diagnosis: ${data.prediction}
            </h4>
            <p class="mb-0">Confidence: ${data.confidence}%</p>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Probability Distribution</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <div class="d-flex justify-content-between mb-1">
                                <span>Normal</span>
                                <span>${data.normal_probability}%</span>
                            </div>
                            <div class="progress" style="height: 20px;">
                                <div class="progress-bar bg-success" style="width: ${data.normal_probability}%"></div>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between mb-1">
                                <span>Pneumonia</span>
                                <span>${data.pneumonia_probability}%</span>
                            </div>
                            <div class="progress" style="height: 20px;">
                                <div class="progress-bar bg-danger" style="width: ${data.pneumonia_probability}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Recommendation</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Risk Level:</strong> 
                            <span class="badge bg-${data.risk_level === 'HIGH' ? 'danger' : data.risk_level === 'MEDIUM' ? 'warning' : 'success'}">
                                ${data.risk_level}
                            </span>
                        </p>
                        <p><strong>Advice:</strong> ${data.recommendation}</p>
                        <p class="text-muted"><small>Analysis completed at ${new Date().toLocaleTimeString()}</small></p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="text-center mt-4">
            <button class="btn btn-primary me-2" onclick="analyzeImage()">
                <i class="fas fa-redo me-1"></i>Analyze Again
            </button>
            <button class="btn btn-success" onclick="clearImage()">
                <i class="fas fa-plus me-1"></i>New Image
            </button>
        </div>
    `;
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector('.card-body').insertBefore(alertDiv, document.querySelector('.card-body').firstChild);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}