let model = null
tf.loadLayersModel("../tfjs-model2/model2.json").then(m => {
    model = m
    console.log("Model:", model)
    document.getElementById("submit-button").disabled = false
    document.getElementById("myfile").disabled = false
})

const image = document.getElementById("myimg")
document.getElementById("myfile").addEventListener("change", (e) => {
    image.src = URL.createObjectURL(e.target.files[0])
    image.style.display = "block"
})

const loadingGif = document.getElementById("mygif")
const containerPred = document.getElementById("container-pred")
document.getElementById("myform").addEventListener("submit", (e) => {
    e.preventDefault()

    if (model !== null && image?.src.startsWith("blob:")) {
        loadingGif.style.display = "block"
        containerPred.style.display = "block"

        const predTensor = tf.tidy(() => {
            const tensor = tf.browser.fromPixels(image)
                .resizeNearestNeighbor([96, 96])
                .toFloat()
                .div(tf.scalar(255))
                .expandDims()

            //tensor.print()
            //console.log(tensor.shape)

            return model.predict(tensor)
        })

        predTensor.data().then(([pred]) => {
            document.getElementById("pred").textContent = (pred > 0.5 ? "Cancer" : "Not Cancer") + ` (${pred.toFixed(5)})`
            if (pred > 0.5) {
                containerPred.classList.remove('gradient-border')
                containerPred.classList.add('gradient-border-cancer')
                document.getElementById('not-cancer-description').style.display = "none"
                document.getElementById('cancer-description').style.display = "block"
                document.getElementById('not-cancer-icon').style.display = "none"
                document.getElementById('cancer-icon').style.display = "block"
            } else {
                containerPred.classList.remove('gradient-border-cancer')
                containerPred.classList.add('gradient-border')
                document.getElementById('cancer-description').style.display = "none"
                document.getElementById('cancer-icon').style.display = "none"
                document.getElementById('not-cancer-icon').style.display = "block"
                document.getElementById('not-cancer-description').style.display = "block"
            }
            document.getElementById("pred").style.color = pred > 0.5 ? "red" : "#A5DC86"
        }).finally(() => { loadingGif.style.display = "none" })

        predTensor.dispose()
        console.log("Memory:", tf.memory())
    }
})