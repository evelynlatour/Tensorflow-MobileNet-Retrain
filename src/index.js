import { ConsoleDisplay } from './lib/display'
import { getXs, oneHot, getYs } from './lib/mobileNet'
import testImage from './images/test.jpg'
import testImage2 from './images/test2.jpg'
import colorful from './images/colorful.jpg'
import { trainingData, dataLabels } from './images'


// new ConsoleDisplay().display()

// formatImage(testImage)

// predictFromTruncated(testImage)

// getXs(trainingData)

// oneHot(3,5).print()

getYs(dataLabels)