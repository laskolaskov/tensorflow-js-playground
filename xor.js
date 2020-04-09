const tf = require('@tensorflow/tfjs')

const xor = (p) => {

    const learningRate = 5
    const totalEpochs = 1000

    const model = tf.sequential()
    const sgdOpt = tf.train.sgd(learningRate)

    const xorTable = tf.tensor2d([
        [1],
        [0],
        [0],
        [1]
    ])

    const inputs = tf.tensor2d([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    model.add(tf.layers.dense({
        name: 'input',
        units: 4,
        activation: 'sigmoid',
        inputDim: 2,
        //batchSize: 100
    }))

    model.add(tf.layers.dense({
        name: 'hidden-1',
        units: 5,
        activation: 'sigmoid',
    }))

    model.add(tf.layers.dense({
        name: 'output',
        units: 1,
        activation: 'sigmoid',
    }))

    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: sgdOpt
    })

    const train = async () => {
        console.log('Training started')
        const history = await model.fit(inputs, xorTable, {
            epochs: totalEpochs,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 10 === 0) {
                        console.log(`Epoch ${epoch} ended !`)
                        console.log(logs)
                    }
                }
            },
            shuffle: true
        })
        console.log('Training ended')
        console.log('training history :: ', history)
    }

    p.setup = async () => {
        p.createCanvas(400, 400)
        p.background(51)

        const res = 40

        const inputData = []

        for (let i = 0; i < p.width / res; i++) {
            for (let j = 0; j < p.height / res; j++) {
                x = p.map(i * res, 0, p.width, 0, 1)
                y = p.map(j * res, 0, p.height, 0, 1)
                inputData.push([x, y])
            }
        }

        await train()

        console.log('Verification:')
        inputs.print()
        xorTable.print()
        model.predict(inputs).print()

        const prediction = model.predict(tf.tensor2d(inputData))
        const result = prediction.dataSync()

        let index = 0
        for (let i = 0; i < p.width / res; i++) {
            for (let j = 0; j < p.height / res; j++) {
                let br = p.map(result[index], 0, 1, 0, 255)
                p.fill(br)
                //p.fill(result[index] * 255)
                p.rect(i * res, j * res, res, res)
                p.fill(255 - br)
                p.textSize(res / 5)
                p.textAlign(p.CENTER, p.CENTER)
                p.text(p.nf(result[index], 1, 3), i * res + res / 2, j * res + res / 2)
                index++
            }
        }

        p.noLoop()
    }
}

module.exports = xor