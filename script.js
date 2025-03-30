const reviews = [
    { name: "Lakhan Singh", role: "Village Leader", text: "The multilingual support made AI KRISHI accessible to my entire community. The pest and disease predictions helped me prevent significant crop losses." },
    { name: "Amit Verma", role: "Farmer", text: "AI KRISHI provided me with the best crop suggestions based on my soil conditions. My yield has increased significantly." },
    { name: "Sunita Sharma", role: "Agricultural Scientist", text: "The AI-driven insights in AI KRISHI help us research and develop better farming strategies for sustainable agriculture." }
];

let currentIndex = 0;
function updateReview(index) {
    document.getElementById("user-name").innerText = reviews[index].name;
    document.getElementById("user-role").innerText = reviews[index].role;
    document.getElementById("review-text").innerText = reviews[index].text;
}
function prevReview() {
    currentIndex = (currentIndex - 1 + reviews.length) % reviews.length;
    updateReview(currentIndex);
    // when we click previous button
    // Showing 1st Review	0	(0 - 1 + 3) % 3 = 2	 3rd Review
    // Showing 3rd Review	2	(2 - 1 + 3) % 3 = 1	 2nd Review
    // Showing 2nd Review	1	(1 - 1 + 3) % 3 = 0	 1st Review

}
function nextReview() {
    currentIndex = (currentIndex + 1) % reviews.length;
    updateReview(currentIndex);
    // when we click next button
    // Showing 1st Review	0	(0 + 1) % 3 = 1	2nd Review
    // Showing 2nd Review	1	(1 + 1) % 3 = 2	3rd Review
    // Showing 3rd Review	2	(2 + 1) % 3 = 0	1st Review
}
document.addEventListener("DOMContentLoaded", () => {
    updateReview(0);
});