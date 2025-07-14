class ChurnPredictionApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;  // Use same origin as the app
        this.init();
        this.checkApiStatus();
    }

    init() {
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.makePrediction();
        });

        // File upload
        document.getElementById('fileUpload').addEventListener('click', () => {
            document.getElementById('csvFile').click();
        });

        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e);
        });

        document.getElementById('batchPredictBtn').addEventListener('click', () => {
            this.makeBatchPrediction();
        });

        // Drag and drop
        const fileUpload = document.getElementById('fileUpload');
        fileUpload.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUpload.style.borderColor = '#667eea';
        });

        fileUpload.addEventListener('dragleave', () => {
            fileUpload.style.borderColor = '#ccc';
        });

        fileUpload.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUpload.style.borderColor = '#ccc';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('csvFile').files = files;
                this.handleFileUpload({ target: { files } });
            }
        });
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            const statusElement = document.getElementById('statusIndicator');
            if (response.ok && data.status === 'healthy') {
                statusElement.textContent = '✅ Online';
                statusElement.style.color = '#28a745';
            } else {
                statusElement.textContent = '❌ Offline';
                statusElement.style.color = '#dc3545';
            }
        } catch (error) {
            const statusElement = document.getElementById('statusIndicator');
            statusElement.textContent = '🔄 Connecting...';
            statusElement.style.color = '#ffc107';
            
            // Retry after 5 seconds
            setTimeout(() => this.checkApiStatus(), 5000);
        }
    }

    async makePrediction() {
        this.showLoading();
        
        try {
            const formData = new FormData(document.getElementById('predictionForm'));
            const customerData = {
                age: parseFloat(formData.get('age')),
                tenure: parseFloat(formData.get('tenure')),
                monthly_charges: parseFloat(formData.get('monthly_charges')),
                total_charges: parseFloat(formData.get('total_charges')),
                contract_length: parseInt(formData.get('contract_length')),
                payment_method: formData.get('payment_method'),
                internet_service: formData.get('internet_service'),
                online_security: formData.get('online_security'),
                tech_support: formData.get('tech_support')
            };

            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(customerData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayPredictionResult(result);

        } catch (error) {
            this.showError(`Prediction failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayPredictionResult(result) {
        const resultsDiv = document.getElementById('predictionResults');
        const predictionCard = document.getElementById('predictionCard');
        const predictionResult = document.getElementById('predictionResult');
        const probabilityFill = document.getElementById('probabilityFill');
        const probabilityText = document.getElementById('probabilityText');
        const riskLevel = document.getElementById('riskLevel');
        const recommendations = document.getElementById('recommendations');

        // Set prediction text
        const willChurn = result.churn_prediction === 1;
        predictionResult.textContent = willChurn ? 
            '⚠️ High Churn Risk' : '✅ Low Churn Risk';
        predictionResult.style.color = willChurn ? '#dc3545' : '#28a745';

        // Set probability bar
        const probability = result.churn_probability * 100;
        probabilityFill.style.width = `${probability}%`;
        probabilityFill.style.backgroundColor = this.getRiskColor(result.risk_level);
        
        probabilityText.textContent = `Churn Probability: ${probability.toFixed(1)}%`;
        
        // Set risk level
        riskLevel.innerHTML = `<strong>Risk Level:</strong> <span style="color: ${this.getRiskColor(result.risk_level)}">${result.risk_level}</span>`;

        // Set card border color
        predictionCard.className = `prediction-card risk-${result.risk_level.toLowerCase()}`;

        // Generate recommendations
        recommendations.innerHTML = this.generateRecommendations(result);

        resultsDiv.style.display = 'block';
    }

    generateRecommendations(result) {
        let recommendations = '<div style="margin-top: 15px;"><strong>💡 Recommendations:</strong><ul style="margin-left: 20px; margin-top: 10px;">';
        
        if (result.risk_level === 'High') {
            recommendations += `
                <li>🎯 Immediate retention campaign required</li>
                <li>📞 Schedule personal contact within 48 hours</li>
                <li>💰 Consider offering special discount or upgrade</li>
                <li>🔄 Review contract terms and service quality</li>
            `;
        } else if (result.risk_level === 'Medium') {
            recommendations += `
                <li>📧 Send targeted email campaign</li>
                <li>🎁 Offer loyalty rewards or benefits</li>
                <li>📞 Proactive customer service call</li>
                <li>📊 Monitor account activity closely</li>
            `;
        } else {
            recommendations += `
                <li>✅ Customer is in good standing</li>
                <li>🌟 Continue providing excellent service</li>
                <li>📈 Consider upselling opportunities</li>
                <li>💌 Include in customer satisfaction surveys</li>
            `;
        }
        
        recommendations += '</ul></div>';
        return recommendations;
    }

    getRiskColor(riskLevel) {
        switch(riskLevel.toLowerCase()) {
            case 'low': return '#28a745';
            case 'medium': return '#ffc107';
            case 'high': return '#dc3545';
            default: return '#6c757d';
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file && file.type === 'text/csv') {
            document.querySelector('#fileUpload p').textContent = `📄 ${file.name} selected`;
            document.getElementById('batchPredictBtn').disabled = false;
        } else {
            this.showError('Please select a valid CSV file');
        }
    }

    async makeBatchPrediction() {
        const fileInput = document.getElementById('csvFile');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select a CSV file first');
            return;
        }

        this.showLoading();

        try {
            // Parse CSV file
            const text = await file.text();
            const customers = this.parseCSV(text);
            
            if (customers.length === 0) {
                throw new Error('No valid customer data found in CSV');
            }

            // Make batch prediction
            const response = await fetch(`${this.apiBaseUrl}/batch_predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(customers)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayBatchResults(result.predictions);

        } catch (error) {
            this.showError(`Batch prediction failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    parseCSV(text) {
        const lines = text.split('\n').filter(line => line.trim());
        if (lines.length < 2) return [];

        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        const customers = [];

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length !== headers.length) continue;

            const customer = {};
            headers.forEach((header, index) => {
                const value = values[index].replace(/"/g, '');
                
                // Convert to appropriate types
                if (['age', 'tenure', 'monthly_charges', 'total_charges'].includes(header)) {
                    customer[header] = parseFloat(value) || 0;
                } else if (header === 'contract_length') {
                    customer[header] = parseInt(value) || 1;
                } else {
                    customer[header] = value;
                }
            });

            // Validate required fields
            if (this.isValidCustomer(customer)) {
                customers.push(customer);
            }
        }

        return customers;
    }

    isValidCustomer(customer) {
        const required = ['age', 'tenure', 'monthly_charges', 'total_charges', 
                         'contract_length', 'payment_method', 'internet_service', 
                         'online_security', 'tech_support'];
        return required.every(field => customer.hasOwnProperty(field));
    }

    displayBatchResults(predictions) {
        const batchResults = document.getElementById('batchResults');
        const batchStatistics = document.getElementById('batchStatistics');

        // Calculate statistics
        const total = predictions.length;
        const highRisk = predictions.filter(p => p.risk_level === 'High').length;
        const mediumRisk = predictions.filter(p => p.risk_level === 'Medium').length;
        const lowRisk = predictions.filter(p => p.risk_level === 'Low').length;
        const avgProbability = predictions.reduce((sum, p) => sum + p.churn_probability, 0) / total;

        // Display statistics
        batchStatistics.innerHTML = `
            <div class="stat-item">
                <div class="stat-number">${total}</div>
                <div class="stat-label">Total Customers</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" style="color: #dc3545">${highRisk}</div>
                <div class="stat-label">High Risk</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" style="color: #ffc107">${mediumRisk}</div>
                <div class="stat-label">Medium Risk</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" style="color: #28a745">${lowRisk}</div>
                <div class="stat-label">Low Risk</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${(avgProbability * 100).toFixed(1)}%</div>
                <div class="stat-label">Avg Churn Risk</div>
            </div>
        `;

        // Create chart
        this.createBatchChart([highRisk, mediumRisk, lowRisk]);

        batchResults.style.display = 'block';
    }

    createBatchChart(data) {
        const canvas = document.getElementById('batchChart');
        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (this.batchChart) {
            this.batchChart.destroy();
        }

        this.batchChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                datasets: [{
                    data: data,
                    backgroundColor: ['#dc3545', '#ffc107', '#28a745'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    title: {
                        display: true,
                        text: 'Risk Distribution'
                    }
                }
            }
        });
    }

    showLoading() {
        document.getElementById('loadingIndicator').style.display = 'block';
        document.getElementById('predictionResults').style.display = 'none';
        document.getElementById('batchResults').style.display = 'none';
        document.getElementById('errorMessage').style.display = 'none';
        document.getElementById('predictBtn').disabled = true;
    }

    hideLoading() {
        document.getElementById('loadingIndicator').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
    }

    showError(message) {
        const errorDiv = document.getElementById('errorMessage');
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        document.getElementById('predictionResults').style.display = 'none';
        document.getElementById('batchResults').style.display = 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChurnPredictionApp();
});