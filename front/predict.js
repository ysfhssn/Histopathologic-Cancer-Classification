let model = null
tf.loadLayersModel("../tfjs-model2/model2.json").then(m => {
    console.log("Model:", m)
    model = m
})

const image = document.getElementById("myimg")
document.getElementById("myfile").addEventListener("change", (e) => {
    image.src = URL.createObjectURL(e.target.files[0])
    image.style.display = "block"
})

const loadingGif = document.getElementById("mygif")
document.getElementById("myform").addEventListener("submit", async (e) => {
    e.preventDefault()

    if (model !== null && image !== null) {
        loadingGif.style.display = "block"

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
            document.getElementById("pred").style.color = pred > 0.5 ? "red" : "green"
            loadingGif.style.display = "none"
        }).catch(err => { console.error(err) })

        predTensor.dispose()
        console.log("Memory:", tf.memory())
    }
})