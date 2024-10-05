import numpy as np

class Linear_Regression():
		
		#initiating the parameter
		def __init__(self, Learning_rate, no_of_iteration):
							self.Learning_rate = Learning_rate
							self.no_of_iteration = no_of_iteration


		def fit(self,X,Y):
			#number of training examples & number of features
			self.m, self.n = X.shape  # for exame we get (30,1) as shape, so self.m=30 and self.n=1(no of rows and no of columns)
			# inititating the weight and bias of our model
			self.w=np.zeros(self.n)# want to create a matrix with n number of columns and all the value in the array should contain 0	
			#let say if  we have many featutres, we can't put self.w as zero for initial value, cause every feature have it's own weight/slope
			self.b=0
			self.X=X
			self.Y=Y


			#implementing gradient descent 
			for i in range(self.no_of_iteration):
				self.update_weight()


		def update_weight(self):

			Y_prediction=self.predict(self.X)#now we need to compare the loss function for different parameters

			#calculating the gradients 
			dw = -(2*(self.X.T).dot(self.Y-Y_prediction))/self.m #self.x.t, here t is doing transpose, which means if ther are 30 row and 1 column, it will converted to 1 row and 30 column
			db=-2 * np.sum(self.Y-Y_prediction)/self.m

			#updating the weights 
			self.w=self.w-self.Learning_rate*dw
			self.b=self.b-self.Learning_rate*db
		
		def predict(self,X ):
			return X.dot(self.w)+self.b #we are including dot because of an arrray
		

