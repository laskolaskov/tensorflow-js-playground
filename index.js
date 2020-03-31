const p5 = require('p5')
const tf = require('@tensorflow/tfjs')

//const randomInt32Arr = (length, max) => Array(length).fill(0).map(_ => Math.floor(Math.random() * max))

const linearRegressionExample = (p) => {

    //data set for X and Y axis
    const xs = []
    const ys = []

    //line function parameters
    let m = tf.scalar(Math.random()).variable()
    let b = tf.scalar(Math.random()).variable()

    //learning rate
    const lr = 0.1
    //optimizer
    const optimizer = tf.train.sgd(lr)

    //f(x) = mx + b
    const f = x => x.mul(m).add(b)

    //loss function
    const loss = (prediction, label) => prediction.sub(label).square().mean()

    p.setup = () => {
        p.createCanvas(400, 400)
    }

    p.draw = () => {
        p.background(0)
        //draw data set
        drawDataPoints()
        //draw line
        tf.tidy(() => {
            drawLine()
        })

        //memory leak check
        console.log('total tensors :: ', tf.memory().numTensors)
    }

    p.mousePressed = () => {
        //add the mouse click as data point
        addDataPoint()
        tf.tidy(() => {
            //train with all data points N times
            for (let i = 0; i < 10; i++) {
                optimizer.minimize(() => loss(f(tf.tensor1d(xs)), tf.tensor1d(ys)))
            }
        })
    }

    //maps pixel coords to range [0, 1]
    const normalize = (x, y) => ({
        a: p.map(x, 0, p.width, 0, 1),
        b: p.map(y, 0, p.height, 1, 0)
    })

    //maps number couples from [0, 1] to pixel coords
    const pixelize = (a, b) => ({
        x: p.map(a, 0, 1, 0, p.width),
        y: p.map(b, 0, 1, p.height, 0)
    })

    const drawDataPoints = () => {
        for (let i = 0; i < xs.length; i++) {
            const { x, y } = pixelize(xs[i], ys[i])
            p.stroke(255)
            p.strokeWeight(8)
            p.point(x, y)
        }
    }

    const drawLine = () => {
        //get the Y line coords for X = 0,1
        const y = f(tf.tensor1d([0, 1])).dataSync()
        //pixelize
        const p1 = pixelize(0, y[0])
        const p2 = pixelize(1, y[1])
        //draw
        p.stroke(255)
        p.strokeWeight(1)
        p.line(p1.x, p1.y, p2.x, p2.y)
    }

    const addDataPoint = () => {
        const { a, b } = normalize(p.mouseX, p.mouseY)
        xs.push(a)
        ys.push(b)
    }
}

//create sketch
const P5 = new p5(linearRegressionExample)