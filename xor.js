const tf = require('@tensorflow/tfjs')

const xor = async (p) => {

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

    console.log(model)

    const history = await model.fit(inputs, xorTable, {
        epochs: totalEpochs,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch} ended !`)
                console.log(logs)
            }
        },
        shuffle: true
    })
    console.log(history)

    inputs.print()
    xorTable.print()
    model.predict(inputs).print()

    p.setup = () => {


    }

    p.draw = () => {

    }

}

module.exports = xor