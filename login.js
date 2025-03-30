const container= document.getElementById('container')
const registerBtn=document.getElementById('register')
const loginBtn=document.getElementById('login')

registerBtn.addEventListener('click', () => {
    // using active we are adding the css and html of active class
  container.classList.add("active")
}
)

loginBtn.addEventListener('click', () => {
    // using active we are removing the css and html of active class
  container.classList.remove("active")
}
)