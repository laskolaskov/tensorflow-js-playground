const p5 = require('p5')
const tf = require('@tensorflow/tfjs')

tf.tensor([1, 2, 3, 4]).print()

const sketch = (p) => {

    p.setup = () => {
        p.createCanvas(400, 400)
        p.background(0)
    }

    p.draw = () => {
        
    }
}

//create sketch
const P5 = new p5(sketch)