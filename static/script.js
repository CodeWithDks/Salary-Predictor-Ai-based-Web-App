// DOM Elements
const form = document.getElementById('predictionForm');
const resultsSection = document.getElementById('results');
const loadingOverlay = document.getElementById('loadingOverlay');
const predictedSalaryElement = document.getElementById('predicted-salary');

// Form inputs
const ageInput = document.getElementById('age');
const genderInput = document.getElementById('gender');
const educationInput = document.getElementById('education');
const experienceInput = document.getElementById('experience');
const jobTitleInput = document.getElementById('job_title');

// Result elements
const resultAge = document.getElementById('result-age');
const resultGender = document.getElementById('result-gender');
const resultEducation = document.getElementById('result-education');
const resultExperience = document.getElementById('result-experience');
const resultJob = document.getElementById('result-job');

// Utility Functions
function showLoading() {
    loadingOverlay.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
    document.body.style.overflow = 'auto';
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <div class="error-content">
            <i class="fas fa-exclamation-triangle"></i>
            <h3>Error</h3>
            <p>${message}</p>
            <button onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add error styles
    errorDiv.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        animation: fadeIn 0.3s ease;
    `;
    
    const errorContent = errorDiv.querySelector('.error-content');
    errorContent.style.cssText = `
        background: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        max-width: 500px;
        margin: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        position: relative;
    `;
    
    const icon = errorContent.querySelector('i.fa-exclamation-triangle');
    icon.style.cssText = `
        font-size: 3rem;
        color: #e53e3e;
        margin-bottom: 20px;
    `;
    
    const title = errorContent.querySelector('h3');
    title.style.cssText = `
        color: #2d3748;
        font-size: 1.5rem;
        margin-bottom: 15px;
        font-weight: 600;
    `;
    
    const text = errorContent.querySelector('p');
    text.style.cssText = `
        color: #718096;
        line-height: 1.6;
        margin-bottom: 25px;
    `;
    
    const closeButton = errorContent.querySelector('button');
    closeButton.style.cssText = `
        position: absolute;
        top: 15px;
        right: 15px;
        background: none;
        border: none;
        font-size: 1.5rem;
        color: #a0aec0;
        cursor: pointer;
        padding: 5px;
        border-radius: 50%;
        transition: all 0.3s ease;
    `;
    
    closeButton.addEventListener('mouseenter', () => {
        closeButton.style.background = '#f7fafc';
        closeButton.style.color = '#2d3748';
    });
    
    closeButton.addEventListener('mouseleave', () => {
        closeButton.style.background = 'none';
        closeButton.style.color = '#a0aec0';
    });
    
    document.body.appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

function formatSalary(salary) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(salary);
}

function animateValue(element, start, end, duration) {
    const startTime = performance.now();
    const startValue = parseFloat(start) || 0;
    const endValue = parseFloat(end);
    const difference = endValue - startValue;
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const currentValue = startValue + (difference * easeOutQuart);
        
        element.textContent = formatSalary(currentValue);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = formatSalary(endValue);
        }
    }
    
    requestAnimationFrame(update);
}

function validateForm() {
    const age = parseInt(ageInput.value);
    const experience = parseInt(experienceInput.value);
    const jobTitle = jobTitleInput.value.trim();
    
    // Age validation
    if (age < 18 || age > 65) {
        throw new Error('Age must be between 18 and 65 years');
    }
    
    // Experience validation
    if (experience < 0 || experience > 50) {
        throw new Error('Years of experience must be between 0 and 50');
    }
    
    // Experience vs Age validation
    if (experience > (age - 16)) {
        throw new Error('Years of experience cannot exceed age minus 16');
    }
    
    // Job title validation
    if (jobTitle.length < 2) {
        throw new Error('Please enter a valid job title');
    }
    
    return true;
}

function displayResults(data) {
    // Update result details
    resultAge.textContent = data.input_data.Age;
    resultGender.textContent = data.input_data.Gender;
    resultEducation.textContent = data.input_data['Education Level'];
    resultExperience.textContent = data.input_data['Years of Experience'] + ' years';
    resultJob.textContent = data.input_data['Job Title'];
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Animate salary with current value
    const currentSalary = parseFloat(predictedSalaryElement.textContent.replace(/,/g, '')) || 0;
    animateValue(predictedSalaryElement, currentSalary, data.prediction, 2000);
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
    }, 500);
}

// Form submission handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    try {
        // Validate form
        validateForm();
        
        // Show loading
        showLoading();
        
        // Prepare form data
        const formData = new FormData(form);
        
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Hide loading
        hideLoading();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred while making the prediction');
        }
        
    } catch (error) {
        hideLoading();
        showError(error.message || 'Please check your input and try again');
    }
});

// Input validation and formatting
ageInput.addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    if (value < 18 || value > 65) {
        e.target.style.borderColor = '#e53e3e';
    } else {
        e.target.style.borderColor = '#e2e8f0';
    }
});

experienceInput.addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    const age = parseInt(ageInput.value);
    
    if (value < 0 || value > 50 || (age && value > (age - 16))) {
        e.target.style.borderColor = '#e53e3e';
    } else {
        e.target.style.borderColor = '#e2e8f0';
    }
});

jobTitleInput.addEventListener('input', (e) => {
    const value = e.target.value.trim();
    if (value.length < 2) {
        e.target.style.borderColor = '#e53e3e';
    } else {
        e.target.style.borderColor = '#e2e8f0';
    }
});

// Add some interactive features
document.addEventListener('DOMContentLoaded', () => {
    // Add floating animation to feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.2}s`;
        card.classList.add('fade-in');
    });
    
    // Add hover effects to form inputs
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.parentElement.style.transform = 'translateY(-2px)';
        });
        
        input.addEventListener('blur', () => {
            input.parentElement.style.transform = 'translateY(0)';
        });
    });
    
    // Add click effect to predict button
    const predictBtn = document.querySelector('.predict-btn');
    predictBtn.addEventListener('click', (e) => {
        // Create ripple effect
        const ripple = document.createElement('span');
        const rect = predictBtn.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
        `;
        
        predictBtn.style.position = 'relative';
        predictBtn.style.overflow = 'hidden';
        predictBtn.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    });
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out forwards;
    }
    
    .error-message {
        animation: fadeIn 0.3s ease;
    }
    
    /* Smooth transitions for all interactive elements */
    * {
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    
    /* Enhanced hover effects */
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
    }
    
    .social-link:hover {
        transform: translateY(-5px) scale(1.1);
    }
    
    .predict-btn:hover {
        transform: translateY(-3px) scale(1.02);
    }
    
    /* Loading animation improvements */
    .loading-overlay {
        animation: fadeIn 0.3s ease;
    }
    
    .spinner {
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Mobile optimizations */
    @media (max-width: 768px) {
        .feature-card:hover {
            transform: translateY(-5px) scale(1.01);
        }
        
        .predict-btn:hover {
            transform: translateY(-2px) scale(1.01);
        }
    }
`;

document.head.appendChild(style);