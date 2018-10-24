import blue1 from './blue/blue1.png'
import blue2 from './blue/blue2.gif'
import blue3 from './blue/blue3.jpg'
import blue4 from './blue/blue4.png'
import blue5 from './blue/blue5.jpg'

import red1 from './red/red1.png'
import red2 from './red/red2.png'
import red3 from './red/red3.jpg'
import red4 from './red/red4.jpg'
import red5 from './red/red5.jpg'

import redTest1 from './test_data/red_test_1.jpg'

const trainingData = [ 
  blue1, blue2, blue3, blue4, blue5,
  red1, red2, red3, red4, red5
]

const dataLabels = [
  'blue', 'blue', 'blue', 'blue', 'blue',
  'red', 'red', 'red', 'red', 'red'
]

const testData = redTest1


export { trainingData, dataLabels, testData }