document.addEventListener('DOMContentLoaded', function () {
    const getBtn = document.getElementById('getRecommendations');
    const resultsDiv = document.getElementById('results');

    getBtn.addEventListener('click', async () => {
        const query = document.getElementById('userQuery').value.trim();
        const model = document.getElementById('modelSelect').value;

        if (!query) {
            alert("Please enter a learning preference!");
            return;
        }

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, model: model })
            });

            const data = await response.json();
            resultsDiv.innerHTML = ''; // Clear previous results

            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(course => {
                    const courseHTML = `
                        <div class="list-group-item mb-3 shadow-sm course-card">
                            <h6>${course.title} <span class="badge bg-success">${course.department}</span></h6>
                            <p>${course.description}</p>
                            <div class="progress mb-2">
                                <div class="progress-bar" role="progressbar" style="width: ${course.relevance}%;" aria-valuenow="${course.relevance}" aria-valuemin="0" aria-valuemax="100">
                                    ${course.relevance}%
                                </div>
                            </div>
                            <button class="btn btn-outline-primary btn-sm save-btn" data-id="${course.course_id}">Save Recommendation</button>
                        </div>`;
                    resultsDiv.insertAdjacentHTML("beforeend", courseHTML);
                });

                // Save functionality
                document.querySelectorAll(".save-btn").forEach(button => {
                    button.addEventListener("click", async (e) => {
                        const courseId = e.target.dataset.id;
                        try {
                            const saveResponse = await fetch("/api/save", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ course_id: courseId, query: query, model: model })
                            });

                            if (saveResponse.ok) {
                                e.target.textContent = "Saved!";
                                e.target.disabled = true;
                            } else {
                                alert("Failed to save recommendation.");
                            }
                        } catch (err) {
                            console.error("Error saving recommendation:", err);
                        }
                    });
                });

            } else {
                resultsDiv.innerHTML = `<p class="text-muted">No recommendations found.</p>`;
            }

        } catch (error) {
            console.error("Error fetching recommendations:", error);
            resultsDiv.innerHTML = `<p class="text-danger">Failed to get recommendations. Try again later.</p>`;
        }
    });
});
