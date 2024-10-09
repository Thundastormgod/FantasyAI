document.addEventListener('DOMContentLoaded', function() {
    const resultDiv = document.getElementById('results');

    // Fetch predicted points when the "Get Predicted Points" button is clicked
    document.getElementById('getPredictedPoints').addEventListener('click', function() {
        resultDiv.innerHTML = '<p class="loading">Fetching predicted points...</p>';
        fetch('/api/predicted-points')  // Make sure the endpoint matches your Flask route
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                displayPredictedPoints(data);
            })
            .catch(error => {
                console.error('Error fetching predicted points:', error);
                resultDiv.innerHTML = `<p class="error">Error fetching predictions: ${error.message}</p>`;
            });
    });

    // Display predicted player points
    function displayPredictedPoints(data) {
        resultDiv.innerHTML = "<h2>Predicted Player Points</h2>";
        if (Array.isArray(data) && data.length > 0) {
            data.forEach(player => {
                resultDiv.innerHTML += `<p>${player.name}: ${player.predicted_points.toFixed(2)} points</p>`;
            });
        } else if (data.message) {
            resultDiv.innerHTML += `<p>${data.message}</p>`;
        } else {
            resultDiv.innerHTML += '<p>No predictions available at this time.</p>';
            console.error('No predictions or unexpected data format:', data);
        }
    }

    // Fetch random team when the "Get Random Team" button is clicked
    document.getElementById('getRandomTeam').addEventListener('click', function() {
        resultDiv.innerHTML = '<p class="loading">Generating random team...</p>';
        fetch('/api/random-team')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                displayRandomTeam(data);
            })
            .catch(error => {
                console.error('Error fetching random team:', error);
                resultDiv.innerHTML = `<p class="error">Error generating random team: ${error.message}</p>`;
            });
    });

    // Display random team
    function displayRandomTeam(data) {
        resultDiv.innerHTML = "<h2>Random Team</h2>";
        if (Array.isArray(data.team)) {
            let tableHTML = '<table><tr><th>Player</th><th>Team</th><th>Position</th><th>Predicted Points</th></tr>';
            data.team.forEach(player => {
                tableHTML += `<tr><td>${player.name}</td><td>${player.team}</td><td>${player.position}</td><td>${player.predicted_points.toFixed(2)}</td></tr>`;
            });
            tableHTML += '</table>';
            resultDiv.innerHTML += tableHTML;
        } else {
            resultDiv.innerHTML += `<p>No team data available.</p>`;
        }
    }
});
