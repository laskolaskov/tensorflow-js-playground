const p5 = require('p5')
const polynomialRegression = require('./polynomialRegression')
const xor = require('./xor')

//const randomInt32Arr = (length, max) => Array(length).fill(0).map(_ => Math.floor(Math.random() * max))

//create sketch
//const P5 = new p5(polynomialRegression)
const P5 = new p5(xor)