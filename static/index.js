const dropArea = document.getElementById("drop-area")
const inputFile = document.getElementById("input-file")
const imageView = document.getElementById("img-view")
const output = document.getElementById("output")

inputFile.addEventListener("change", uploadImage)


async function uploadImage() {
    img = inputFile.files[0]
    let imglink = URL.createObjectURL(img)
    imageView.style.backgroundImage = `url(${imglink})`
    imageView.textContent = ''
    output.innerHTML = 'Please Wait for a moment....'

    const formData = new FormData();
    formData.append("file",img)
    var response= await fetch('http://localhost:8000/predict',{
        method:'POST',
        body:formData
    })

    const res=await response.json()
    output.innerHTML = `Predicted_Age : ${res.age}`
    
}   

dropArea.addEventListener("dragover",(e)=>{
    e.preventDefault();
    
})
dropArea.addEventListener("drop",(e)=>{
    e.preventDefault();
    inputFile.files=e.dataTransfer.files
    uploadImage()
})