const init = (async function main () {

    const model = await tf.loadLayersModel("../tfjs-model2/model2.json")
    console.log("Model:", model)

    const image = document.getElementById("myimg")
    document.getElementById("myfile").addEventListener("change", (e) => {
        image.src = URL.createObjectURL(e.target.files[0])
        image.style.display = "block"
    })

    const loadingGif = document.getElementById("mygif")
    const containerPred = document.getElementById("container-pred")
    document.getElementById("myform").addEventListener("submit", async (e) => {
        e.preventDefault()
        loadingGif.style.display = "block"
        containerPred.style.display = "block"

        if (image !== null) {
            const tensor = tf.browser.fromPixels(image)
                .resizeNearestNeighbor([96, 96])
                .toFloat()
                .div(tf.scalar(255))
                .expandDims()

            tensor.print()
            console.log(tensor.shape)

            const pred = await model.predict(tensor).data()
            console.log(pred)

            loadingGif.style.display = "none" // TODO: loading gif not showing up ...

            // BEST THRESHOLD: 0.64063287
            document.getElementById("pred").textContent = pred[0] > 0.5 ? "Cancer" : "Not Cancer"
            if(document.getElementById("pred").textContent == "Cancer"){
                containerPred.classList.remove('gradient-border');
                containerPred.classList.add('gradient-border-cancer');
                document.getElementById('not-cancer-description').style.display = "none";
                document.getElementById('cancer-description').style.display = "block";
                document.getElementById('not-cancer-icon').style.display = "none";
                document.getElementById('cancer-icon').style.display = "block";
            } else {
                containerPred.classList.remove('gradient-border-cancer');
                containerPred.classList.add('gradient-border');
                document.getElementById('cancer-description').style.display = "none";
                document.getElementById('cancer-icon').style.display = "none";
                document.getElementById('not-cancer-icon').style.display = "block";
                document.getElementById('not-cancer-description').style.display = "block";
            }
            document.getElementById("pred").style.color = pred[0] > 0.5 ? "red" : "green"
        }
    })
})()
