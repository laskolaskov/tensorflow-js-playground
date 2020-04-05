const tf = require('@tensorflow/tfjs')

const polynomialRegression = (p) => {

    //data set for X and Y axis
    const xs = []
    const ys = []

    //polynomial function parameters
    let a = tf.scalar(p.random(-1, 1)).variable()
    let b = tf.scalar(p.random(-1, 1)).variable()
    let c = tf.scalar(p.random(-1, 1)).variable()
    //learning rate
    const lr = 0.1
    //optimizer
    const optimizer = tf.train.sgd(lr)

    //f(x) = ax + b - linear regression
    //const f = x => x.mul(a).add(b)
    //f(x) = ax^2 + bx + c
    const f = x => tf.mul(a, x.square()).add(tf.mul(b, x)).add(c)

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
            for (let i = 0; i < 100; i++) {
                optimizer.minimize(() => loss(f(tf.tensor1d(xs)), tf.tensor1d(ys)))
            }
        })
    }

    //maps pixel coords to range [-1, 1]
    const normalize = (x, y) => ({
        a: p.map(x, 0, p.width, -1, 1),
        b: p.map(y, 0, p.height, 1, -1)
    })

    //maps number couples from [-1, 1] to pixel coords
    const pixelize = (a, b) => ({
        x: p.map(a, -1, 1, 0, p.width),
        y: p.map(b, -1, 1, p.height, 0)
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
        //get regular samples of X in [-1, 1]
        const xRange = []
        for(let i = -1; i <= 1; i+= 0.05) {
            xRange.push(i)
        }
        //get the polynomial solutions Y for X in [-1, 1]
        const ys = f(tf.tensor1d(xRange)).dataSync()
        //draw
        p.stroke(255)
        p.strokeWeight(1)
        p.noFill()
        p.beginShape()
        ys.forEach((el, i) => {
            //pixelize
            const point = pixelize(xRange[i], el)
            p.vertex(point.x, point.y)
        })
        p.endShape()
    }

    const addDataPoint = () => {
        const { a, b } = normalize(p.mouseX, p.mouseY)
        xs.push(a)
        ys.push(b)
    }
}

module.exports = polynomialRegression