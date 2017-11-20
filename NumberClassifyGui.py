#NeuralNetworkGui.py

from tkinter import *
import numpy as np

from HelperClasses import *
from VectorNeuralNetwork import *
from DataFormatFunctions import *
from BreadthFirst import *
from HolesFeature import *
from LongestLines import *
from Features import *


def drawRectangle(index, prob, probCanvas):
	canvasWidth = 400
	canvasHeight = 100

	rectangleWidth = canvasWidth/10

	x0 = index*rectangleWidth
	x1 = (index+1)*rectangleWidth

	y0 = ((1-prob)*canvasHeight)
	y1 = canvasHeight

	probCanvas.create_rectangle(x0, y0, x1, y1, fill='blue')


def drawNumClasses(index, probCanvas):
	canvasWidth = 400
	canvasHeight = 100+20

	rectangleWidth = canvasWidth/10

	x = index*rectangleWidth + (.5*rectangleWidth)
	y = canvasHeight-10

	probCanvas.create_text(x, y, text=str(index))


def drawProbRectangles(probVector, probCanvas):
	probCanvas.delete("all")

	for i, prob in enumerate(probVector):
		drawNumClasses(i, probCanvas)
		drawRectangle(i, prob, probCanvas)



def scaleDownCoords(scale, drawn_coords):

	scaled_down_coords = []

	for coords in drawn_coords:
		coords.getMNISTIndex()
		scaledCoords = coords.getScaledDown(scale)
		scaled_down_coords.append(scaledCoords)

	return scaled_down_coords



def translateToNumber(neuralNet, scale, drawn_coords, resultBox, probDistBox, probCanvas):
	imageVector = createImageVector(scale, drawn_coords)

	selectedFeaturesObj = neuralNet.getSelectedFeaturesObj()

	featureVector = createFeatures(imageVector, selectedFeaturesObj, silent=True)

	resultVector = neuralNet.feedForward(featureVector[:,0])
	probStr = createReadableOutputVector(resultVector)

	resultValue = np.argmax(resultVector)

	resultBox.delete('1.0', END)
	resultBox.insert(INSERT, resultValue)

	probDistBox.delete('1.0', END)
	probDistBox.insert(INSERT, probStr)

	drawProbRectangles(resultVector, probCanvas)


def createImageVector(scale, drawn_coords):

	scaled_coords = scaleDownCoords(scale, drawn_coords)

	fattenList = []

	for coords in scaled_coords:

		if (coords.x > 0):
			fattenList.append(Coords(coords.x-1,coords.y))

		if (coords.y < 26):
			fattenList.append(Coords(coords.x,coords.y+1))



	full_coords = scaled_coords + fattenList

	imageVector = np.zeros((784, 1))

	for coords in full_coords:
		drawn_index = coords.getMNISTIndex()
		imageVector[drawn_index] = 1

	return imageVector


def click(click_obj, scale, drawn_coords, canvas):

	x, y = click_obj.x, click_obj.y

	drawn_coords.append(Coords(x,y))

	canvas.create_oval((x,y,x,y), width=scale*3, fill="black")


def mousemotion(motion_obj, scale, drawn_coords, canvas):
	x, y = motion_obj.x, motion_obj.y

	if((0 <= x < IMG_WIDTH*scale) and (0 <= y < 28*IMG_HEIGHT)):
		drawn_coords.append(Coords(x,y))
	
		canvas.create_oval((x,y,x,y), width=scale*3, fill="black")


def clear(canvas, drawn_coords):
	canvas.delete("all")
	drawn_coords.clear()




def runGui(netName):

	neuralNet = loadNeuralNet(netName)


	scale = 16
	
	drawn_coords = []
	
	root = Tk()
	canvasFrame = Frame(root)
	canvasFrame.pack()

	drawLabel = Label(canvasFrame, text="Draw Digit Below")
	drawLabel.pack(side=TOP)



	canvas = Canvas(canvasFrame, width = (28*scale), height = (28*scale), bg='white')
	canvas.pack(side=TOP)
	

	buttonFrame = Frame(root)


	identifyButton = Button(buttonFrame, text='Identify',
			command=lambda: translateToNumber(neuralNet, scale, drawn_coords, resultBox, probDistBox, probCanvas))
	identifyButton.grid(row=0, column=0)
	buttonFrame.grid_columnconfigure(0, pad=100)

	clearButton = Button(buttonFrame, text='Clear', command=lambda: clear(canvas, drawn_coords))
	clearButton.grid(row=1, column=0)

	quitButton = Button(buttonFrame, text='Quit', command=root.quit)
	quitButton.grid(row=0, column=2)
	buttonFrame.grid_columnconfigure(2, pad=100)

	resultLabel = Label(buttonFrame, text="Network Guess")
	resultLabel.grid(row=0, column=1)

	resultBox = Text(buttonFrame, height=1, width=2)
	resultBox.grid(row=1, column=1)

	buttonFrame.pack()

	probDistFrame = Frame(root)

	probDistBox = Text(probDistFrame, height=3, width=50)
	probDistBox.pack(side=BOTTOM)



	probCanvas = Canvas(probDistFrame, width = 400, height = 120, bg='white')
	probCanvas.pack()

	probDistFrame.pack()


	canvas.bind("<Button-1>", lambda event: click(event, scale, drawn_coords, canvas))
	
	canvas.bind("<B1-Motion>", lambda event: mousemotion(event, scale, drawn_coords, canvas))
	
	root.bind("<Return>", lambda event: translateToNumber(neuralNet, scale, drawn_coords, resultBox, probDistBox, probCanvas))



	root.mainloop()


def main():

	if (len(sys.argv) < 2):
		print("Using default network")
		netName = "UnnamedNetwork"
	else:
		netName = sys.argv[1]

	runGui(netName)


if __name__ == "__main__":
	main()
	