"""
Module made by Rhys Thayer from 15 October 2025 to 15 January 2026

This module is meant to give tools for multilinear algebra and linear algebra. 

This module is currently capable of making tensors, matrices, and vectors, and can perform scalar multiplication, tensor contraction, traces, take 
tensor products, transpose tensors and matrices, change tensor index, add and subract tensors,  edit and access elements within a tensor, iteratively 
go through each coordinate of a tensor, determine if a nested list can be turned into a tensor, find the shape of a tensor, and return a readable 
string that details the information about the tensor. 

This module can multiply vectors by scalars, take the cross product of two vectors in R3, take the dot product, find the length of a vector, find the 
angle between vectors, find the component of one vector along another vector, find the projection of one vector onto another, and check to see if a 
set of vectors are linearly independant. 

This module can multiply matrices by scalars, vectors, and other matrices. It can also find the determinant of a matrix, find the inverse of a matrix, 
transpose a matrix, raise a matrix to any integer power, raise e to the power of a matrix, and solve a system of equations. 

This Module CANNOT find eigenvalues or eigenvectors. 

Run Module_alias.info() to learn more details

Classes:
    Tensor: A class to create and manipulate tensors
    Matrix: A class to create and manipulate matrices, subclass of Tensor
    Vector: A class to create and manipulate vectors, subclass of Tensor

Functions:
    info: A function where the user can find specific information about the module
"""

__author__ = "Rhys Thayer"

from random import randint
from random import random
from math import acos
from math import factorial

class Tensor:
    """
    Made by Rhys Thayer

    This class is used to create and manipulate tensors.

    Attributes:
        shape (list): A list of non zero integers representing the shape of the tensor and whether an index is contravarient or covariant.
        tensor (nested list): A nested list containing the elements of the tensor.
    """

    @staticmethod
    def depth(ordered_set:list):
        """
        This Function finds the depth of a nested list.\n
        This is used for validating tensors.

        :param ordered_set: nested list to find the depth of, NOT a tensor
        :return: integer representing the depth of the nested list
        """
        #List case, depth is one plus the max depth of its elements
        if isinstance(ordered_set,list):
            top_depth = 0
            for i in range(len(ordered_set)):
                working_depth = Tensor.depth(ordered_set[i])
                if working_depth > top_depth:
                    top_depth = working_depth
            output = 1 + top_depth
        #Non-list base case, depth is zero
        else:
            output = 0
        return output

    @staticmethod
    def valid_tensor(nest:Tensor|list):
        """
        This Function is used to verify that a nested list can be a valid tensor.

        :param nest: is a list or Tensor to be verified
        :return: boolean, True if valid tensor, False if not
        """
        #If a Tensor is provided, extract the nested list
        if isinstance(nest, Tensor):
            nest = nest.tensor
        #List case, check if all elements are numbers or all elements are lists of the same depth and length
        if isinstance(nest, list):
            #Check if all elements are numbers
            if all(((isinstance(i, int)) or (isinstance(i, float))) for i in nest):
                return True
            #Check if all elements are lists of the same depth and length
            if all((isinstance(i,list) and (len(i) == len(nest[0])) and (Tensor.depth(i) == Tensor.depth(nest[0]))) for i in nest):

                    for i in nest:
                        if all((Tensor.valid_tensor(i) == True) for i in nest):
                            return True
                        else:
                            return False
            #If neither case is true, it is not a valid tensor
            else:
                return False
        #Base case, check if it is a number
        elif isinstance(nest, int) or isinstance(nest, float) or isinstance(nest, complex):
            return True
        #If neither case is true, it is not a valid tensor
        else:
            return False

    @staticmethod
    def find_shape(tensor:list):
        """
        Finds the shape of a tensor. Only works for valid tensors.\n
        Shape is returned as a list of integers, where each integer is the length of the sublist,\n
        from outermost to innermost. resulting shape will only have contravariant indices (positive integers).\n
        Negative integers represent covariant indices, and are not represented here.

        :Param tensor: is a nested list representing a valid tensor
        :return: a list of integers representing the shape of the tensor and the indices
        """
        #Check to make sure input is valid
        if not Tensor.valid_tensor(tensor):
            raise TypeError("Provided tensor is not a valid tensor")
        
        #Prime shape
        shape = []

        def recursive_shape(tensor,shape):
            #Each time a list is encountered, add its length to shape and go deeper
            if isinstance(tensor,list):
                shape = recursive_shape(tensor[0], shape)
                shape.insert(0, len(tensor)) 
            return shape
        
        shape = recursive_shape(tensor, shape)
        return shape

    @staticmethod
    def loop_through_all_coordinates(shape:Tensor|list):
        """
        Generator that loops through all coordinates of a tensor with given shape. \n
        Yields each coordinate as a list of integers, starting from [0,0,...,0] to the maximum coordinate. \n
        Negative dimensions in shape are treated as positive.

        :param shape: list of integers representing the shape of the tensor or a Tensor
        :return: list of intgers representing the current coordinate for an element in the tensore
        """
        if isinstance(shape, Tensor):
            shape = shape.shape[:]
        #Take care of negatives
        #shape = [abs(i) for i in shape]
        for i in range(len(shape)):
            if shape[i] < 0:
                shape[i] *= -1

        #Prime the coordinates
        coordinates = []
        for i in range(len(shape)):
            coordinates.append(0)

        #Loop through all coordinates
        while True:

            yield list(coordinates)

            # increment from the end like an odometer
            i = len(coordinates) - 1
            while i >= 0:
                coordinates[i] += 1
                if coordinates[i] < shape[i]:
                    break
                coordinates[i] = 0
                i -= 1
            # stop condition
            if i < 0:
                break

    def __str__(self):
        """
        Converts the Tensor to a readable string format.
        
        :param self: a Tensor object
        :return: string representing the Tensor in readable format
        """
        string = f"Order: {len(self.shape)} \nShape: {self.shape}\nTensor:\n"
        if len(self.shape) == 0:
            string += str(self.tensor)
        elif len(self.shape) == 1:
            string += str(self.tensor)
        elif len(self.shape) == 2:
            for i in range(abs(self.shape[0])):
                    string += f"    col {i}: {self.access_element([i])}\n"
        elif len(self.shape) > 2:
            thing = self.shape[:]
            thing.pop()
            other_thing = thing.pop()
            for i in Tensor.loop_through_all_coordinates(thing):
                string += f"    Slice {i}\n"
                for j in range(other_thing):
                    coords = i[:]
                    coords.append(j)
                    string += f"        col {j}: {self.access_element(coords)}\n"
        return string

    def __init__(self, shape:list = None, variety:str = "custom", tensor:Tensor|int|float|complex = None, min:int = -9, max:int = 9):
        """
        Create a tensor! \n
        shape is a list of the length of the array in each dimension, \n
        Whether the dimension is - or + specifies whether it is an upper or lower index. \n
        tensor can be inputed manually with tensor = [nested list]

        :param shape: list of non zero integers representing the shape of the tensor and whether an index is contravarient or covariant.
        :param variety: string representing the variety of tensor to be made, options are "zero", "identity", "diagonal", "randomint", "random", and "custom". Default is "custom".
        :param tensor: nested list representing the elements of the tensor
        :param min: integer representing the minimum value for randomint variety.
        :param max: integer representing the maximum value for randomint variety.
        :return: a Tensor object
        """

        def _make_zero_tensor(shape, rank):
            #Makes a tensor of all zeros
            x = []
            if rank == len(shape):
                return 0
            for i in range(abs(shape[rank])):
                x.append(_make_zero_tensor(shape, rank+1))
            return x

        def _make_identity_tensor(shape, rank, position):
            #Makes an identity tensor, not sure how useful this is for non-matrix tensors
            x = []
            if rank == len(shape):
                if len(set(position)) > 1:
                    return 0
                else:
                    return 1
            for i in range(abs(shape[rank])):
                position[rank] = i      #if acting weird, origionally was rank-1
                x.append(_make_identity_tensor(shape, rank+1, position))
            return x

        def _make_diagonal_tensor(shape, rank, position):
            #Makes a diagonal tensor, asks user for input for diagonal elements
            x = []
            if rank == len(shape):
                if len(set(position)) > 1:
                    return 0
                else:
                    print("What number at position ", end="")
                    for i in range(len(position)):
                        if i == len(position)-1:
                            print(position[len(position)-i-1],end="")
                        else:
                            print(f"{position[len(position)-i-1]}, ",end="")
                element = input(f"? ")
                try:
                    element = int(element)
                except:
                    try:
                        element = float(element)
                    except:
                        try:
                            element = complex(element)
                        except:
                            raise TypeError("Element must be int, float, or complex number")
                return element
            for i in range(abs(shape[rank])):
                position[rank] = i      #if acting weird, origionally was rank-1
                x.append(_make_diagonal_tensor(shape, rank+1, position))
            return x

        def _make_random_int_tensor(shape, rank, min, max):
            #Makes a tensor of random integers between min and max, default -9 to 9
            x = []
            if rank == len(shape):
                return randint(min,max)
            for i in range(abs(shape[rank])):
                x.append(_make_random_int_tensor(shape, rank+1, min, max))
            return x

        def _make_random_tensor(shape, rank):
            #Makes a tensor of random floats between 0 and 1
            x = []
            if rank == len(shape):
                return random()
            for i in range(abs(shape[rank])):
                x.append(_make_random_tensor(shape, rank+1))
            return x

        def _make_custom_tensor(shape, rank, position):
            #Makes a tensor with user input for each element
            x = []
            if rank == len(shape):
                print("What number at position ", end="")
                for i in range(len(position)):
                    if i == len(position)-1:
                        print(position[len(position)-i-1],end="")
                    else:
                        print(f"{position[len(position)-i-1]}, ",end="")
                element = input(f"? ")
                try:
                    element = int(element)
                except:
                    try:
                        element = float(element)
                    except:
                        try:
                            element = complex(element)
                        except:
                            raise TypeError("Element must be int, float, or complex number")
                return element
            for i in range(abs(shape[rank])):
                position[rank-1] = i
                x.append(_make_custom_tensor(shape, rank+1, position))
            return x

        #Sort through how the Tensor will be made
        if shape is None:
            #Senario where only a valid tensor is provided
            if Tensor.valid_tensor(tensor):
                self.tensor = tensor
                self.shape = Tensor.find_shape(tensor)
                for i in range(len(self.shape)):
                    self.change_index(i)
            #Cannot make a valid Tensor without shape or valid tensor
            else:
                raise TypeError("provided tensor is not a valid tensor")
        elif isinstance(shape, Tensor):
            #Senario where a Tensor is being copied
            self.shape = shape.shape
            self.tensor = shape.tensor
        elif isinstance(shape,list):
            #Verify shape is valid
            for i in range(len(shape)):
                if type(shape[i]) != int or shape[i] == 0:
                    raise TypeError("shape must be a list containing only nonzero integers")
            #Senario where shape and variety are provided but no tensor
            if tensor is None:
                self.shape = shape
                rank = 0
                #Primer for position to keep track of current position in recursive functions
                position = []
                for i in range(len(shape)):
                    position.append(0)

                if variety == "zero":
                   self.tensor = _make_zero_tensor(shape, rank)
                elif variety == "identity":
                    self.tensor = _make_identity_tensor(shape, rank, position)
                elif variety == "diagonal":
                    self.tensor = _make_diagonal_tensor(shape, rank, position)
                elif variety == "randomint":
                    self.tensor = _make_random_int_tensor(shape, rank, min, max)
                elif variety == "random":
                    self.tensor = _make_random_tensor(shape, rank)
                elif variety == "custom":
                    self.tensor = _make_custom_tensor(shape, rank, position)
                else:
                    raise TypeError("Cannot make Tensor, not supported valid variety")
            #Senario where shape and tensor are provided
            elif Tensor.valid_tensor(tensor):
                actual_shape = Tensor.find_shape(tensor)
                if len(shape) == len(actual_shape):
                    for i in range(len(shape)):
                        if abs(shape[i]) != abs(actual_shape[i]):
                            raise TypeError("Shape provided is not the same shape as the tensor provided")
                    self.shape = shape
                    self.tensor = tensor
                else:
                    raise TypeError("Shape provided is not the same shape as the tensor provided")
            else:
                raise TypeError("provided tensor is not a valid tensor")
        else:
            print(shape,variety,tensor)
            raise TypeError("Cannot make Tensor with provided arguments. Run Tensor.info() for help with this module. ")

    def access_element(self:Tensor, coordinates:list):
        """
        Access an element within the tensor at the provided coordinates.
        
        :param self: Tensor object to be accessed
        :param coordinates: list of indices used to search the nested list that is the tensor
        :return: element at the provided coordinates
        """
        if not(isinstance(coordinates, list) and all(isinstance(i, int) for i in coordinates)):
            raise TypeError("Coordinates must be a list of integers")
        if len(coordinates) > len(self.shape):
            raise TypeError("Coordinates have more dimensions than the Tensor")
        if coordinates == []:
            return self.tensor
        else:
            #Make sure coordinates are in range
            for i in range(len(coordinates)):
                if coordinates[i] > abs(self.shape[i]):
                    raise IndexError("Coordinate out of range for tensor")
                if coordinates[i] < 0:
                    coordinates[i] *= -1
            #Find element at coordinates
            current_level = self.tensor
            
            for coordinate in coordinates:
                # Sequentially access the next dimension
                current_level = current_level[coordinate]
            return current_level

    def edit_element(self, coordinates:list, element:int|float|complex|Tensor|list, *, element_permissions=False):
        """
        Method to chonge an element within the tensor at the provided coordinates
        
        :param self: Tensor object to be accessed
        :param coordinates: list of indices used to search the nested list that is the tensor
        :param element: element to replace the current element with
        :param element_permissions: used internally to bypass shape checking when replacing an element with a tensor product
        """
        #Make sure coordinates are properly formatted
        if not(isinstance(coordinates, list) and all(isinstance(i, int) for i in coordinates)):
            raise TypeError("Coordinates must be a list of integers")
        #Make sure coordinates are not too long
        if len(coordinates) > len(self.shape):
            raise TypeError("Coordinates have more dimensions than the Tensor")
        #Make sure coordinates are in range
        for i in range(len(coordinates)):
            if coordinates[i] > abs(self.shape[i])-1:
                raise IndexError("Coordinate out of range for tensor")
            if coordinates[i] < 0:
                coordinates[i] *= -1


        #If the element is a Tensor, extract the nested list
        if isinstance(element, Tensor):
            element = element.tensor
        #Access the current element to be changed
        #Check that new element is valid and same shape as current element
        if element_permissions == False:
            current_level = self.tensor
            for coordinate in coordinates:
                current_level = current_level[coordinate]

            if Tensor.find_shape(current_level) != Tensor.find_shape(element):
                raise TypeError("New element is not the same shape as the element being changed")

        #Prime the position and tensor for recursive function
        position = list(coordinates)
        if isinstance(self.tensor, list):
            tensor = list(self.tensor)
        elif isinstance(self.tensor, int):
            tensor = int(self.tensor)
        elif isinstance(self.tensor, float):
            tensor = float(self.tensor)

        def _change_element(self, coordinates, element, position, tensor):
            if position == []:
                element_shape = list(self.shape)
                for i in range(len(coordinates)):
                    element_shape.pop()
                return element
            else:
                index = position.pop(0)
                tensor[index] = _change_element(self, coordinates, element, position, tensor[index])
            return tensor

        self.tensor = _change_element(self, coordinates, element, position, tensor)

    def __add__(tensor_0:Tensor, tensor_1:int|float|complex):
        """
        Tensor Addition \n
        Adds two tensors of the same shape together element wise \n
        Every index in the shape must be the same for both tensors \n
        can also add a scalar to a scalar tensor. 

        :param tensor_0: first Tensor object to be added
        :param tensor_1: second Tensor object to be added or a scalar
        :return: a new Tensor object that is the sum of the two tensors
        """

        #recursive function to add each element pair wise
        def _add_element(self, tensor_0, tensor_1, coordinates, position=None, tensor=None):
            if len(coordinates) > len(self.shape):
                raise TypeError("Coordinates have more dimensions than the Tensor")
            if position == None:
                position = list(coordinates)
            if tensor == None:
                tensor = self.tensor
            if position == []:
                coordinates_1 = list(coordinates)
                coordinates_2 = list(coordinates)
                element = (tensor_0.access_element(coordinates_1)) + (tensor_1.access_element(coordinates_2))
                return element
            else:
                index = position.pop(0)
                tensor[index] = _add_element(self, tensor_0, tensor_1, coordinates, position, tensor[index])
                return tensor

        #account for when numbers of different classes are added
        if isinstance(tensor_1, int):
            x = int(tensor_1)
            tensor_1 = Tensor([], "zero")
            tensor_1.tensor = int(x)
        if isinstance(tensor_1, float):
            x = float(tensor_1)
            tensor_1 = Tensor([], "zero")
            tensor_1.tensor = float(x)
        if isinstance(tensor_1, complex):
            x = complex(tensor_1)
            tensor_1 = Tensor([], "zero")
            tensor_1.tensor = complex(x)
        if tensor_0.shape != tensor_1.shape:
            raise TypeError("These tensors cannot be added, they have different indices")
        new_tensor = Tensor(list(tensor_0.shape),"zero")

        shape = list(tensor_0.shape)

        #add each element pair wise
        for i in Tensor.loop_through_all_coordinates(shape):
            new_tensor.tensor = _add_element(new_tensor, tensor_0, tensor_1, i)

        return new_tensor

    def __radd__(tensor_0:Tensor, tensor_1:int|float|complex):
        #If the left operand is a scalar, flip the operation around so the scalar is on the right
        if isinstance(tensor_1, int) or isinstance(tensor_1, float) or isinstance(tensor_1, complex):
            return tensor_0+tensor_1

    def __sub__(tensor_0:Tensor, tensor_1:Tensor|int|float|complex):
        """
        Tensor Subtraction

        :param tensor_0: first Tensor object to be subtracted from
        :param tensor_1: second Tensor object to be subtracted or a scalar
        :return: a new Tensor object that is the difference of the two tensors
        """
        if isinstance(tensor_1, int) or isinstance(tensor_1, float) or isinstance(tensor_1, complex):
            return tensor_0 + (-1*tensor_1)
        elif isinstance(tensor_1, Tensor):
            return tensor_0 + (tensor_1%-1)

    def __rsub__(tensor_0:Tensor, tensor_1:int|float|complex):
        #If the left operand is a scalar, flip the operation around so the scalar is on the right and multiply by -1
        if isinstance(tensor_1, int) or isinstance(tensor_1, float) or isinstance(tensor_1, complex):
            return (tensor_0 - tensor_1)%-1

    def __neg__(self):
        """
        Multiply Tensor by -1
        
        :param self: tensor to be multiplied by -1
        """
        return self%-1

    def __mod__(tensor_0:Tensor, tensor_1:Tensor|int|float|complex):
        """
        Tensor Product

        :param tensor_0: first Tensor object to be multiplied
        :param tensor_1: second Tensor object to be multiplied or a scalar
        :return: a new Tensor object that is the tensor product of the two tensors or scalar
        """
        #Case of scalar multiplication
        if isinstance(tensor_1, int) or isinstance(tensor_1,float) or isinstance(tensor_1,complex):
            # scalar multiplication, just multiply each element by the scalar
            new_tensor = Tensor(tensor_0.shape,"zero")
            for coordinates in Tensor.loop_through_all_coordinates(list(tensor_0.shape)):
                old_val = tensor_0.access_element(coordinates)
                new_tensor.edit_element(coordinates, tensor_1 * old_val)
                #tensor_0.edit_element(coordinates, (tensor_1 * (tensor_0.access_element(coordinates))))
            return new_tensor
        
        #Case of tensor product
        elif isinstance(tensor_1, Tensor):
            final_shape = list(tensor_0.shape) + list(tensor_1.shape)
            final_tensor = Tensor(final_shape, "zero")
            #for each element in tensor 0, multiply the element by the entire tensor 1 and place it back where the element is located
            for coordinates in Tensor.loop_through_all_coordinates(list(tensor_0.shape)):
                new_tensor = tensor_1 % tensor_0.access_element(coordinates)
                final_tensor.edit_element(coordinates, new_tensor.tensor, element_permissions=True)
            #correct the shape
            final_tensor.shape = final_shape
            return final_tensor

    def __rmod__(tensor_0:Tensor, tensor_1:int|float|complex):
        #If the left operand is a scalar, flip the operation around so the scalar is on the right
        if isinstance(tensor_1, int) or isinstance(tensor_1, float) or isinstance(tensor_1,complex):
            return tensor_0%tensor_1

    def __mul__(tensor_0:Tensor, scalar:int|float|complex):
        """
        Multiply a tensor by a scalar
        
        :param tensor_0: tensor to be scaled
        :param scalar: scalar to multiply tensor by
        :return: scaled tensor
        """
        if isinstance(scalar, int) or isinstance(scalar, float) or isinstance(scalar, complex):
            return tensor_0%scalar
        else:
            return NotImplemented

    def __rmul__(tensor_0:Vector, scalar:int|float|complex):
        if isinstance(scalar, int) or isinstance(scalar, float) or isinstance(scalar, complex):
            tensor_0%scalar
        else:
            return NotImplemented

    @staticmethod
    def contract(tensor_0:Tensor, tensor_1:Tensor, index_0:int, index_1:int):
        """
        Contract two tensors along specified indices
        
        :param tensor_0: First tensor to be contracted
        :param tensor_1: second tensor to be contracted
        :param index_0: index of first tensor to be contracted
        :param index_1: index of second tensor to be contracted
        """
        if isinstance(tensor_0.tensor, int) or isinstance(tensor_0.tensor, float) or isinstance(tensor_1,complex):
            raise TypeError("Scalars cannot be contracted")
        if tensor_0.shape[index_0] != -1*tensor_1.shape[index_1]:
            raise TypeError("Cannot contract indices of different lengths and cannot contract two upper or two lower indices")
        index_1 += len(tensor_0.shape)
        if index_0 == index_1:
            raise TypeError("Cannot contract the same index")
        return (tensor_0%tensor_1).trace(index_0, index_1)

    def trace(tensor_0, index_1:int, index_2:int):
        """
        Tensor Contraction for a single tensor along two specified indices

        :param tensor_0: Tensor to be contracted
        :param index_1: first index to be contracted
        :param index_2: second index to be contracted
        :return: new Tensor object that is the contraction of the two indices
        """

        #Catch errors for invalid contraction
        if index_1 == index_2:
            raise TypeError("Cannot contract the same index")
        if tensor_0.shape[index_1] != -1*tensor_0.shape[index_2]:
            raise TypeError("Cannot contract indices of different lengths and cannot contract two upper or two lower indices")

        #Remove the contracted indices from the new tensor's shape
        new_tensor_shape = list(tensor_0.shape)
        if index_1 > index_2:
            new_tensor_shape.pop(index_1)
            new_tensor_shape.pop(index_2)
        elif index_2 > index_1: 
            new_tensor_shape.pop(index_2)
            new_tensor_shape.pop(index_1)

        #Scaffolding for new tensor
        new_tensor = Tensor(new_tensor_shape, "zero")

        #Loop through each diagonal element and add them to the new tensor
        for i in range(abs(tensor_0.shape[index_1])):
            diagonal_element = Tensor(new_tensor_shape,"zero")
            #Set up diagonal element
            for j in Tensor.loop_through_all_coordinates(list(new_tensor_shape)):
                shape_coords = list(j)
                if index_1 > index_2:
                    shape_coords.insert(index_2, i)
                    shape_coords.insert(index_1, i)
                elif index_2 > index_1:
                    shape_coords.insert(index_1, i)
                    shape_coords.insert(index_2, i)
                #j.reverse()
                diagonal_element.edit_element(j,tensor_0.access_element(shape_coords))

            new_tensor = new_tensor+diagonal_element

        return new_tensor

    def transpose(tensor_0, index_1:int, index_2:int):
        """
        Transpose a tensor by swapping two indices

        :param tensor_0: Tensor to be transposed
        :param index_1: first index to be swapped
        :param index_2: second index to be swapped
        :return: new Tensor object that is the transposed tensor
        """
        #Catch errors for invalid transpose
        if index_1 == index_2:
            raise TypeError("Cannot transpose the same index")

        old_tensor_shape = list(tensor_0.shape)
        new_tensor_shape = list(tensor_0.shape)
        new_tensor_shape[index_1], new_tensor_shape[index_2] = old_tensor_shape[index_2], old_tensor_shape[index_1]
        print("Old shape:", old_tensor_shape)
        print("New shape:", new_tensor_shape)

        #Transpose the tensor
        new_tensor = Tensor(new_tensor_shape, "zero")
        for old_coords in Tensor.loop_through_all_coordinates(old_tensor_shape):
            new_coords = list(old_coords)
            new_coords[index_1], new_coords[index_2] = old_coords[index_2], old_coords[index_1]
            value = tensor_0.access_element(old_coords)
            new_tensor.edit_element(new_coords, value)

        return new_tensor

    def change_index(self, index:int):
        """
        Flips an index in .shape from contravarient to covariont and vice versa
        
        :param self: Tensor to change index of
        :param index: which index to change
        """
        #Negative numbers are covariant indices. Positive numbers are contravariant indices (standard vectors)."
        self.shape[index] *= -1

    def __int__(self):
        """
        Converts a scalar tensor to an integer

        :return: integer of the order 0 tensor
        """
        if self.shape != []:
            raise TypeError("Only scalar tensors can be converted to integers")
        return int(self.access_element([]))
    
    def __float__(self):
        """
        Converts a scalar tensor to a float

        :return: float of the order 0 tensor
        """
        if self.shape != []:
            raise TypeError("Only scalar tensors can be converted to floats")
        return float(self.access_element([]))

    def __complex__(self):
        """
        Converts a scalar tensor to a complex number

        :return: complex number of the order 0 tensor
        """
        if self.shape != []:
            raise TypeError("Only scalar tensors can be converted to complex numbers")
        return complex(self.access_element([]))

    def __iter__(self):
        """
        Iterator for tensors

        :return: yields each element in the tensor one by one
        """
        for i in Tensor.loop_through_all_coordinates:
            yield self.access_element(i)

    def __eq__(self, tensor_1:Tensor):
        """
        Equality operator for tensors

        :param self: First tensor to compare
        :param tensor_1: Second tensor to compare
        :return: True if tensors elements and indices are the same, False otherwise
        """
        if len(self.shape) == 0 and (isinstance(tensor_1, int) or isinstance(tensor_1, float) or isinstance(tensor_1, complex)):
            return True
        else:
            if (self.shape == tensor_1.shape) and (self.tensor == tensor_1.tensor):
                return True
            else:
                return False

    def __len__(self):
        """
        Order of the tensor

        :param self: Tensor to find the order of
        :return: integer representing the order of the tensor
        """
        return len(self.shape)



class Vector(Tensor):
    """
    Made by Rhys Thayer

    This class is used to create and manipulate vectors. \n
    Subclass of Tensor. \n
    Attributes:
        shape (list): A list of non zero integers representing the shape of the tensor and whether an index is contravarient or covariant.
        tensor (list): A list containing the elements of the vector.
    """
    def __init__(self, shape:list|int = None, variety:str = "custom", *, tensor:Tensor|list|int|float|complex = None, min:int = -9, max:int = 9):
        """
        shape is a list of one integer specifying number of elements. \n
        1st dimension is the number of elements, do not worry about indices here. \n
        vector can be inputed manually with tensor = [list]\n
        shape can be a Tensor object to convert it to a Vector

        :param shape: nonzero integor or list of one non zero integer representing the number of elements in the vector or a Tensor to be converted to a Vector. Alternatively, a list representing the elements of the vector con be passed. 
        :param variety: string representing the variety of vector to be made, options are "zero", "identity", "diagonal", "randomint", "random", and "custom". Default is "custom".
        :param tensor: list representing the elements of the vector
        :param min: integer representing the minimum value for randomint variety.
        :param max: integer representing the maximum value for randomint variety.
        :return: a Vector object
        """
        if isinstance(shape, Tensor):
            if len(shape.shape) != 1:
                raise TypeError("Cannot convert Tensor to Vector, Tensor is not one dimensional")
            super().__init__(shape.shape, tensor=shape.tensor, min=min, max=max)
        elif tensor != None:
            #shape = Tensor.find_shape(tensor)
            shape = [shape]
            super().__init__(shape, variety, tensor=tensor, min=min, max=max)
        elif isinstance(shape, int):
            shape = [shape]
            super().__init__(shape, variety, tensor=tensor, min=min, max=max)
        elif isinstance(shape, list):
            this_shape = Tensor.find_shape(shape)
            super().__init__(this_shape, variety, tensor=shape, min=min, max=max)
        else:
            raise TypeError("Cannot make Vector with provided arguments. Run Tensor.info() for help with this module. ")

    def __str__(self):      #Needs to have proper formatting for large tensors
        string = f"Order: {len(self.shape)} \nShape: {self.shape}\nVector:\n    {self.tensor}"
        return string

    def __mul__(vector_0:Vector, scalar:Vector|int|float|complex):
        """
        Multiply vector by scalar or multiply a covector by a vector to get dot product
        
        :param vector_0: vector or covector to be multiplied
        :param scalar: scalar to multiply vector by or a Vector to take the dot product with
        """
        if isinstance(scalar, int) or isinstance(scalar, float) or isinstance(scalar, complex):
            return super().__mod__(scalar)
        elif isinstance(scalar, Vector):
            if (vector_0.shape[0] < 0) and (scalar.shape[0] > 0):
                return Vector.dot_product(vector_0, scalar)
            else:
                raise TypeError("Dot product requires a covariant vector to be multiplied by a contravariant vector")
        else:
            return NotImplemented
    def __rmul__(vector_0:Vector, scalar:int|float|complex):
        if isinstance(scalar, int) or isinstance(scalar, float) or isinstance(scalar, complex):
            return super().__mod__(scalar)
        else:
            return NotImplemented

    def access_element(self, coordinates):
        """
        Access an element within the vector at the provided coordinates.
        
        :param self: Tensor object to be accessed
        :param coordinates: integer or length one list of indices used to search the vector
        :return: element at the provided coordinates
        """
        if isinstance(coordinates, int):
            coordinates = [coordinates]
        return super().access_element(coordinates)

    def edit_element(self, coordinates, position=None, tensor=None):
        """
        method to chonge an element within the tensor at the provided coordinates.

        :param self: Vector object to be accessed
        :param coordinates: integer or length one list of indices used to search the vector
        :param element: element to replace the current element with
        """
        if isinstance(coordinates, int):
            coordinates = [coordinates]
        return super().edit_element(coordinates, position, tensor)

    def change_index(self):
        """
        Changes Contravectors to Covectors and vice versa
        
        :param self: Vector to change index of
        """
        return super().change_index(0)

    def dot_product(vector_0, vector_1):
        """
        Dot product of two vectors
        
        :param vector_0: first vector in dot product
        :param vector_1: second vector in dot product
        :return: scalar result of dot product
        """
        if ((vector_0.shape[0] < 0) and (vector_1.shape[0] < 0)) or ((vector_0.shape[0] > 0) and (vector_1.shape[0] > 0)):
            vector_0.shape[0] *= -1
        return Tensor.contract(vector_0, vector_1, 0, 0).access_element([])

    def cross_product(vector_1, vector_2):
        """
        Cross product of two three dimensional vectors
        
        :param vector_1: first three dimensional vector in cross product
        :param vector_2: second three dimensional vector in cross product
        :return: three dimensional vector that is the cross product of the two input vectors
        """
        if (len(vector_1.tensor) != 3) or (len(vector_2.tensor) != 3):
            raise TypeError("Only 3 dimensional vectors")
        return Vector([(vector_1.tensor[1] * vector_2.tensor[2]) - (vector_2.tensor[1] * vector_1.tensor[2]),(vector_1.tensor[0] * vector_2.tensor[2]) - (vector_2.tensor[0] * vector_1.tensor[2]),(vector_1.tensor[0] * vector_2.tensor[1]) - (vector_2.tensor[0] * vector_1.tensor[1])])

    def __abs__(self):
        """
        Magnitude of a vector

        :param self: vector to find magnitude of
        :return: scalar magnitude of the vector
        """
        return (sum(i**2 for i in self.tensor))**(0.5)
    
    def magnitude(self):
        """
        Magnitude of a vector

        :param self: vector to find magnitude of
        :return: scalar magnitude of the vector
        """
        return abs(self)

    def angle_between(vector_0, vector_1):
        """
        Finds the angle between two vectors in radians
        
        :param vector_0: first vector in angle calculation
        :param vector_1: second vector in angle calculation
        :return: angle between the two vectors in radians
        """
        return acos(Vector.dot_product(vector_0, vector_1) / (abs(vector_0) * abs(vector_1)))

    def projection(vector_0, vector_1):
        """
        finds the projection of vector_0 onto vector_1
        
        :param vector_0: vector being projected
        :param vector_1: vector being projected onto
        :return: vector that is the projection of vector_0 onto vector_1
        """
        new_vector = vector_1*(Vector.dot_product(vector_0, vector_1)/((abs(vector_1))**2))
        if new_vector.shape[0] < 0:
            new_vector.shape[0] *= -1
        return new_vector

    def component(vector_0, vector_1):
        """
        Finds the scalar component of vector_0 in the direction of vector_1
        
        :param vector_0: vector being measured
        :param vector_1: vector being measured against
        :return: scalar component of vector_0 in the direction of vector_1
        """
        return Vector.dot_product(vector_0, vector_1)/(abs(vector_1))

    @staticmethod
    def linear_independance(*vectors:Vector):
        """
        Checks if a set of vectors are linearly independent
        
        :param vectors: at least two Vector objects to check for linear independance
        :type vectors: Vector
        :return: True if the vectors are linearly independent, False otherwise
        """
        #This fuction could be unreliable with complex numbers
        #check number of arguements
        if (len(vectors) == 0) or (len(vectors) == 1):
            raise TypeError("Cannot find linear independance of zero or one vactor")
        #Check type of arguements
        for vector in vectors:
            if not isinstance(vector, Vector):
                raise TypeError("Not all arguments are vectors")
        #Check that all the vectors are the same length
        if all(len(vectors[0]) == len(vectors[i]) for i in range(len(vectors))):
            #See if a simple square matrix can be made
            if len(vectors) == len(vectors[0].tensor):
                #make vectors into a square matrix
                vector_matrix = []
                for vector in vectors:
                    vector_matrix.append(vector.tensor)
                matrix = Matrix(tensor=vector_matrix)
                if abs(matrix) == 0:
                    return False
                else:
                    return True
            #If there are more vectors than dimensions, the vectors are always linearly dependant
            elif len(vectors) > len(vectors[0].tensor):
                print("b")
                return False
            #If there are more dimensions than vectors, turn them into a matrix and take the determinant of the transpose multiplied by the matrix
            elif len(vectors) < len(vectors[0].tensor):
                vector_matrix = []
                for vector in vectors:
                    vector_matrix.append(vector.tensor)
                matrix = Matrix(tensor=vector_matrix)
                transpose_matrix = matrix.transpose()
                if abs(matrix*transpose_matrix) == 0:
                    return False
                else:
                    return True
        else:
            raise TypeError("Not all vectors are the same dimensions")



class Matrix(Tensor):
    """
    Made by Rhys Thayer

    This class is used to create and manipulate tensors. \n

    Attributes:
        shape (list): A list of non zero integers representing the shape of the tensor and whether an index is contravarient or covariant.
        tensor (nested list): A nested list containing the elements of the tensor. Columns are the innermost list.
    """

    def __init__(self, arg_1=None, arg_2=None, arg_3=None, *, tensor:Tensor|list|int|float|complex = None, min:int = -9, max:int = 9):
        """
        shape is a list of two integers specifying number of rows and columns. \n
        1st dimension is rows (m), 2nd dimension is columns (n), do not worry about indices here.\n
        matrix can be inputed manually with tensor = [nested list]\n
        shape can be a Tensor object to convert it to a Matrix

        :param arg_1: list of two non zero integers representing the number of rows and columns in the matrix or a Tensor to be converted to a Matrix. Alternatively, a nonzero integer representing the number of rows can be passed.
        :param arg_2: string representing the variety of matrix to be made, options are "zero", "identity", "diagonal", "randomint", "random", and "custom". Default is "custom". Alternatively, a nonzero integer representing the number of columns can be passed.
        :param arg_3: optional string representing the variety of matrix to be made, options are "zero", "identity", "diagonal", "randomint", "random", and "custom". Default is "custom".
        :param tensor: nested list representing the elements of the matrix
        :param min: integer representing the minimum value for randomint variety.
        :param max: integer representing the maximum value for randomint variety.
        :return: a Matrix object
        """

        if isinstance(arg_1, Tensor):
            if len(arg_1.shape) != 2:
                raise TypeError("Cannot convert Tensor to Matrix, Tensor is not two dimensional")
            shape = list(arg_1.shape)
            if shape[0] > 0:
                shape[0] *= -1
            if shape[1] < 0:
                shape[1] *= -1
            super().__init__(shape, tensor=arg_1.tensor)
        elif tensor != None:
            if not Tensor.valid_tensor(tensor):
                raise TypeError("provided tensor is not a valid tensor")
            if len(Tensor.find_shape(tensor)) != 2:
                raise TypeError("provided tensor is not a valid matrix, must be two dimensional")
            shape = Tensor.find_shape(tensor)
            if shape[0] > 0:
                shape[0] *= -1
            if shape[1] < 0:
                shape[1] *= -1
            super().__init__(shape, tensor=tensor)
        elif isinstance(arg_1, int) and isinstance(arg_2, int):
            if arg_3 is None:
                super().__init__([ -1*abs(arg_1), abs(arg_2)], variety="custom", min=min, max=max)
            elif isinstance(arg_3, str):
                super().__init__([ -1*abs(arg_1), abs(arg_2)], variety=arg_3, min=min, max=max)
        elif isinstance(arg_1, list):
            if arg_1[0] > 0:
                arg_1[0] *= -1
            if arg_1[1] < 0:
                arg_1[1] *= -1
            if arg_2 is None:
                super().__init__(arg_1, variety="custom", min=min, max=max)
            elif isinstance(arg_2, str):
                super().__init__(arg_1, variety=arg_2, min=min, max=max)
        else:
            raise TypeError("Cannot make Matrix with provided arguments. Run Tensor.info() for help with this module. ")

    def __str__(self):
        string = f"Order: {len(self.shape)} \nShape: {self.shape}\nMatrix:\n"
        for i in range(abs(self.shape[0])):
            string += f"    col {i}: {self.access_element([i])}\n"

        return string

    def __mul__(tensor_0:Matrix, tensor_1:Tensor|Matrix|Vector|int|float|complex):
        """
        For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix. This means -1*tensor_0.shape[0] == tensor_1.shape[0]\n
        The resulting matrix has the number of rows of the first and the number of columns of the second matrix.

        :param tensor_0: first Matrix object to be multiplied
        :param tensor_1: scalar, Vector object, or second Matrix object to be multiplied
        :return: a new Matrix or Vector object that is the product of the first matrix and either scalar, vector, or second matrix
        """
        if isinstance(tensor_1, int) or isinstance(tensor_1, float) or isinstance(tensor_1, complex):
            return Matrix(tensor_0%tensor_1)
        elif isinstance(tensor_1, Vector):
            if -1*tensor_0.shape[0] != tensor_1.shape[0]:
                raise TypeError("Cannot multiply these, number of columns in first matrix must equal number of rows in second")
            return Vector(Tensor.contract(tensor_0, tensor_1, 0, 0))
        elif isinstance(tensor_1, Matrix):
            if -1*tensor_0.shape[0] != tensor_1.shape[1]:
                raise TypeError("Cannot multiply these, number of columns in first matrix must equal number of rows in second")
            return Matrix(Tensor.contract(tensor_0, tensor_1, 1, 0))
        else:
            print("fjhsz")
            return NotImplemented

    def __rmul__(tensor_0:Matrix, tensor_1:int|float|complex):
        if isinstance(tensor_1, int) or isinstance(tensor_1, float) or isinstance(tensor_1, complex):
            return Matrix.__mul__(tensor_0, tensor_1)
        else:
            return NotImplemented

    def __abs__(matrix):
        """
        Determinant of a square matrix. May be unreliable with complex numbers because of numerical error\n
        from python's handling of complex numbers, and slow for large matrices due to recursive nature.

        :param matrix: Matrix object to find the determinant of
        :return: scalar determinant of the matrix
        """

        def determinant(matrix):
            if len(matrix) == 1 and len(matrix[0]) == 1:
                if isinstance(matrix[0][0], int) or isinstance(matrix[0][0], float) or isinstance(matrix[0][0], complex):
                    return matrix[0][0]
            elif isinstance(matrix, list):
                det = 0
                for i in range(len(matrix)):
                    det += ((-1)**i)*matrix[0][i]*determinant([row[:i] + row[i+1:] for row in matrix[1:]])
                return det



        if matrix.shape[0] != -1*matrix.shape[1]:
            raise TypeError("Determinant only defined for square matrices")
        return determinant(matrix.tensor)

    def determinant(matrix):
        """
        Determinant of a square matrix. May be unreliable with complex numbers because of numerical error\n
        from python's handling of complex numbers, and slow for large matrices due to recursive nature.

        :param matrix: Matrix object to find the determinant of
        :return: scalar determinant of the matrix
        """
        return abs(matrix)

    def __pow__(matrix, power:int):
        """
        Matrix Exponentiation\n
        Raises a square matrix to an integer power\n
        Only defined for square matrices and for integer powers\n
        Raising to 0 returns the identity matrix\n
        Raising to a negative power returns the inverse matrix raised to the absolute value of that power

        :param matrix: Matrix object to be raised to a power
        :param power: integer power to raise the matrix to
        :return: new Matrix object that is the original matrix raised to the specified power
        """
        if matrix.shape[0] != -1*matrix.shape[1]:
            raise TypeError("Matrix exponentiation only defined for square matrices")
        if not isinstance(power, int):
            raise TypeError("Matrix exponentiation only defined for integer powers")
        if power == 0:
            return Tensor((matrix.shape[0], matrix.shape[1]), "identity")
        elif power < 0:
            return (matrix.inverse())**(-power)
        else:
            result = Matrix(matrix)
            for i in range(power-1):
                result = result * matrix
            return result

    def transpose(matrix):
        """
        Transposes a matrix
        
        :param matrix: matrix to be transposed
        :return: transposed matrix
        """
        return Matrix(super().transpose(0,1))

    def inverse(matrix):
        """
        Finds the inverse of a square matrix using the adjugate method\n
        May be unreliable with complex numbers because of numerical error due to how python handles complex numbers\n
        returns TypeError if the matrix does not have an inverse

        :param matrix: Description
        :return: inverse of matrix
        """
        #Check the determinant of the matrix
        if abs(matrix) == 0:
            raise TypeError("This matrix does not have an inverse")
        #I would rather redefine the determinant here than to make a new matrix for each minor
        def determinant(matrix):
            if len(matrix) == 1 and len(matrix[0]) == 1:
                if isinstance(matrix[0][0], int) or isinstance(matrix[0][0], float) or isinstance(matrix[0][0], complex):
                    return matrix[0][0]
            elif isinstance(matrix, list):
                det = 0
                for i in range(len(matrix)):
                    det += ((-1)**i)*matrix[0][i]*determinant([row[:i] + row[i+1:] for row in matrix[1:]])
                return det

        #scaffolding for the inverse matrix
        cofactor_matrix = Matrix(matrix.shape, "zero")
        #For whatever reason, I cannot clone matrix any other way than this for-loop
        for element in Tensor.loop_through_all_coordinates(list(matrix.shape)):
            cofactor_matrix.edit_element(element, matrix.access_element(element))
        #priming minor
        minor = [i[:] for i in matrix.tensor]
        for element in Tensor.loop_through_all_coordinates(list(cofactor_matrix.shape)):
            #Only way I was able to clone minor without messing up matrix
            minor = [i[:] for i in matrix.tensor]
            #remove the elements that share either index with the current element
            for j in range(len(minor)):
                minor[j].pop(element[1])
            minor.pop(element[0])
            #turning minor into an element to put in the cofactor matrix
            minor = ((-1)**(element[0]+element[1]))*determinant(minor)
            cofactor_matrix.edit_element([element[1], -1*element[0]], minor)
        #Formula for an inverse matrix
        return Matrix((1/abs(matrix))%cofactor_matrix)

    def exp(matrix):
        """
        Uses the Taylor series expansion to raise e to the power of a square matrix\n
        goes up to the 20th term in the series for approximation
        
        :param matrix: matrix to raise e to the power of
        :return: e raised to the power of the matrix
        """
        if matrix.shape[0] != -1*matrix.shape[1]:
            raise TypeError("e can only be raised to the power of square matrices")
        output = Matrix(matrix.shape, "identity")
        for i in range(1,20):
            output+= (1/factorial(i))*(matrix**i)
        return output

    def solve_system(matrix, vector):
        """
        Solves a system of equations represented by Ax = b, where A is the matrix, x is the vector of variables, and b is the result vector.

        :param matrix: Coefficient matrix A
        :param vector: Result vector b
        :return: Solution vector x
        """
        if not isinstance(vector, Vector):
            raise TypeError("Second argument must be a Vector")
        inverse_matrix = matrix.inverse()
        return inverse_matrix * vector

    #Eigenvalues            Not implimented
    #diagonalize            Not implimented 

def info():
    """
    This function directs the user to their desired information about this module. 
    """
    user_input = ""
    while True:
        print("Enter q anytime to quit, or b to go back")
        print("Enter a number:\n1: About this module\n2: Tensors\n3: Matrices\n4: Vectors")
        user_input = input("> ")
        #about this module
        if user_input == "1":
            print("""
Module made by Rhys Thayer

This module is meant to give tools for multilinear algebra and linear algebra. 

This module is currently capable of making tensors, matrices, and vectors, and can perform scalar multiplication, tensor contraction, traces, take 
tensor products, transpose tensors and matrices, change tensor index, add and subract tensors,  edit and access elements within a tensor, iteratively 
go through each coordinate of a tensor, determine if a nested list can be turned into a tensor, find the shape of a tensor, and return a readable 
string that details the information about the tensor. 

This module can multiply vectors by scalars, take the cross product of two vectors in R3, take the dot product, find the length of a vector, find the 
angle between vectors, find the component of one vector along another vector, find the projection of one vector onto another, and check to see if a 
set of vectors are linearly independant. 

This module can multiply matrices by scalars, vectors, and other matrices. It can also find the determinant of a matrix, find the inverse of a matrix, 
transpose a matrix, raise a matrix to any integer power, raise e to the power of a matrix, and solve a system of equations. 

This Module CANNOT find eigenvalues or eigenvectors. 
                """)        
            print("\nenter q to quit, or b to go back to the previous menu")
            user_input = input("> ")
            while (user_input != "q") and (user_input != "b"):
                print("Did not understand user input")
                user_input = input("> ")
            if user_input == "b":
                continue
            elif user_input == "q":
                break
        #Tensors
        elif user_input == "2":
            while True:
                print("Enter q anytime to quit, or b to go back")
                print("Enter a number:\n1: What is a Tensor?\n2: Defining a Tensor & representation\n3: Tensor operations\n4: Other functions")
                user_input = input("> ")
                if user_input == "1":
                    print("""
A tensor is an array of numbers, that come in different orders depending on the number of dimensions to the array. An order zero tensor is simply a 
scalar ann order one tensor is a vector, and an order two tensor is a matrix. This can keep going to higher order tensors. Tensors map to other 
tensors through contraction. Tensors are used in differential geometry and physics for things like moment of inertia and general relativity. 
Wikipedia has more on this: https://en.wikipedia.org/wiki/Tensor""")
                    print("enter q to quit, or b to go back to the previous menu")
                    user_input = input("> ")
                    while (user_input != "q") and (user_input != "b"):
                        print("Did not understand user input")
                        user_input = input("> ")
                    if user_input == "b":
                        continue
                    elif user_input == "q":
                        break
                elif user_input == "2":
                    print("""
In this module tensors have two attributes, .tensor and .shape. .tensor is represented as a multidimensional list. .shape is represented as a list of 
nonzero integers where the magnitude of each element is how many dimensions are in that index and whether it is negative or positive indicate a lower 
or upper index respectively. The magnitude of the first index in .shape is how many elements are in the outermost part of the multidimensional list. 
Conversely, the magnitude of the last index in .shape is how many elements are in the innermost lists. 

There are multiple ways to define a tensor in this module. The first is to provide a shape and a variety. shape is the first positional arguement and 
variety is the second. By default, variety is "custom". The varieties are:
- "randomint": fills the tensor with random integers between -9 and 9 by default, can be changed with the kwargs min= and max=
- "random": fills the tensor with a random float between 0 and 1
- "zero": every element is a zero
- "custom": user provides each element with the input() function. Coordinates for each element are indicated.
- "diagonal": every element where each coordinate for the element is the same takes user input with input() and all other elements are zero. 
              Coordinates for each element are indicated.
- "identity": every element where each coordinate for the element is the same is one and all other elements are zero. Coordinates for each element are 
              indicated.
This will look like module_alias.Tensor(shape, variety, min=, max=) or module_alias.Tensor([-3,3,2]], "randomint", min=0, max=10)

The second way to define a tensor in this module is with a shape and a nested list. The nested list must be passed through the kwarg tensor=
This will look like module_alias.Tensor(shape, tensor=nested_list) or module_alias.Tensor([-2,2], tensor=[[1,2],[3,4]])
A tensor can also be defined by passing a nested list through the kwarg tensor= without providing a shape. The shape will be found automatically 
and all indices will be lower. Lastly, a tensor can be passed. 

When a tensor is printed, the order, shape, and the multidimensional list representing the tensor are shown. The tensor may be of a higher dimensional 
list than the two dimensional screen, so it will show a two dimensional cross section one at a time with a slice specified. The slice is a list of 
coordinates where the first coordinate corrosponds to the which element in the outer most list in .tensor, and if the the coordinates are passed to 
the access_element() function  it will return a two dimensional list. The slice shown will show each column in the two dimensional list with it's 
index, and each column is an innermost list.""")
                    print("enter q to quit, or b to go back to the previous menu")
                    user_input = input("> ")
                    while (user_input != "q") and (user_input != "b"):
                        print("Did not understand user input")
                        user_input = input("> ")
                    if user_input == "b":
                        continue
                    elif user_input == "q":
                        break
                elif user_input == "3":
                    while True:
                        print("Enter q anytime to quit, or b to go back")
                        print("Enter a number:\n1: Tensor Product or 'multiplication'\n2: Addition/Subraction\n3: Contraction\n4: Trace\n5: Transpose\n6: Change index")
                        user_input = input("> ")
                        if user_input == "1":
                            print("""
The tensor product can take any two tensors and multiply every element by every other element. This is equivalent to replacing each element in the 
first tensor with the second tensor scaled by the origional element. The indices of the new tensor is the first tensor's concatenated with the second 
tensor, and the order of the new tensor is the sum of the order of the two tensors. This operation is not communitive, but is is associative. This 
module has a reprogrammed __mod__ dundermethod uses % to indicate a tensor product. For example, to return the tensor product of tensor_1 and tensor_2, it would be tensor_1%tensor_2""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "2":
                            print("\nTensor addition and subtraction can only be performed between two tensors with the same shape and indices it adds two tensors element wise. \nTo use it simply use tensor_1+tensor_2 or tensor_1-tensor_2. The negative sign can also turn a tensor negative elementwise")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "3":
                            print("""
Tensor contraction takes two tensors and two indices, and returns a tensor of order of the sum of the two tensors minus 2. This operation is 
equivalent to taking the tensor product of the two tensors and then the trace of the new tensor with appropriate indices. Matrix-vector 
multiplication, matrix-matrix multiplication, and the dot product are all special cases of tensor contraction. The indices that you contract two 
tensors by must have the same number of dimensions and one index must be upper and the other lower. In this module the shape and indices are 
represented with a list accessed with tensor.shape where the magnitude of each element is how many dimensions are in that index and whether it is 
negative or positive indicate a lower or upper index respectively. 
To use, type Module_alias.Tensor.contraction(tensor_1, tensor_2, index_1, index_2) 
where each index points to it's respective tensor, not the new tensor""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "4":
                            print("""
The trace method takes two indices, one upper and one lower of the same number of dimensions, then sums each array of elements that share the same 
contravariant and covarient index. This reduces the order of a tensor by two. Example use would be tensor.trace(index_1, index_2)""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "5":
                            print("""
The Transpose method takes any indices of the tensor and swaps them. All associated elements in the tensor get swapped. 
Example would be tensor.transpose(index_1, index_2)""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "6":
                            print("\nThe change_index method takes an index and swaps it's index from upper to lower or lower to upper")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        if user_input == "q":
                            break
                        elif user_input == "b":
                            pass
                        else:
                            print("Did not understand user input")
                            continue
                        break
                    if user_input == "b":
                        continue
                    else:
                        break
                elif user_input == "4":
                    while True:
                        print("Enter q anytime to quit, or b to go back")
                        print("Enter a number:\n1: depth\n2: valid_tensor\n3: find_shape\n4: loop_through_all_coordinates\n5: access_element\n6: edit_element\n7: change_index\n8: casting as a number\n9: iteration\n10: equivalence\n11: order or length")
                        user_input = input("> ")
                        if user_input == "1":
                            print("""
The depth static method finds how nested a nested list is recursively. If the passed argument is not a list, the returned depth is zero. 
This is used in verifying a nested list can be turned into a tensor.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "2":
                            print("""
The valid_tensor static method  checks to see if a nested list can be turned into a tensor. This means the all elements are either numbers or lists 
and the depth of each list within a list is the same.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "3":
                            print("""
The static method find_shape finds the shape of a nested list that can be turned into tensors. The shape will have all upper indices""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "4":
                            print("""
loop_through_all_coordinates is a generator that gives the coordinates, or indices for each element in a tensor. 
This static method takes a tensor or list of integers. """)
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "5":
                            print("""
This method returns an element in a tensor if provided a list of integer coordinates. This method can also return lists of lists in the tensor.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "6":
                            print("""
This method takes a list of integer coordinates and replaces the element at those coordinate with a number or nested list. """)
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "7":
                            print("\nThe change_index method takes an indexand swaps it's index from upper to lower or lower to upper")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "8":
                            print("""
An order zero tensor is a scalar, or an ordinary number. To make the exchange between the two easier, an order zero tensor can be cast as a complex number, float, or integer.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "9":
                            print("""
If ever you need a iterable, __iter__ borrows from loop_through_all_coordinates to loop through all possible coordinates in a tensor. """)
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "10":
                            print("""
The dunder method __eq__ is reprogrammed to chekc if the shape and tensor of a tensor object are the same.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "11":
                            print("""
The dunder method __len__ is reprogrammed to return the order of a tensor, which is given by the length of tensor.shape""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        if user_input == "q":
                            break
                        elif user_input == "b":
                            pass
                        else:
                            print("Did not understand user input")
                            continue
                        break
                    if user_input == "b":
                        continue
                    else:
                        break
                if user_input == "q":
                    break
                elif user_input == "b":
                    pass
                else:
                    print("Did not understand user input")
                    continue
                break
            if user_input == "b":
                pass
            else:
                break
        #Matrices
        elif user_input == "3":
            while True:
                print("Enter q anytime to quit, or b to go back")
                print("Enter a number:\n1: Defining a Matrix & representation\n2: Matrix operations\n3: Limitations")
                user_input = input("> ")
                if user_input == "1":
                    print("""
Matrices in this module are a subclass of Tensors. Therefore, they inherit all the properties and methods of Tensors and have some matrix specific 
functions. Their shape is of the form [-n, m] where m is the number of rows and n is the number of columns. The first index is always negative, 
indicating a covariant index. 
                          
Creating an instance of a matrix is the same as creating any other tensor, only there is no need to worry over whether the indices are upper or lower, 
and you can alternatively make one with the template module_alias.Matrix(#rows, #columns, variety)
                          
When a matrix is printed, the order, shape, and the two dimensional list representing the tensor are shown. The matrix shown will show each column 
in the two dimensional list with it's index, and each column as a list.""")

                    print("\nenter q to quit, or b to go back to the previous menu")
                    user_input = input("> ")
                    while (user_input != "q") and (user_input != "b"):
                        print("Did not understand user input")
                        user_input = input("> ")
                    if user_input == "b":
                        continue
                    elif user_input == "q":
                        break
                elif user_input == "2":
                    while True:
                        print("Enter q anytime to quit, or b to go back")
                        print("Enter a number:\n1: Multiplication\n2: Determinant\n3: Inverse\n4: Solving systems of equations\n5: Transpose\n6: Exponentiation\n7: e to the power of a matrix")
                        user_input = input("> ")
                        if user_input == "1":
                            print("""
The dunder method __mul__ is reprogrammed so that * can be used to multiply matrices by scalars, vectors, and other matrices. Multiplication calls 
the Tensor class's tensor contraction method. """)
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "2":
                            print("""
The dunder method __abs__ is reprogrammed so that abs(matrix) returns the determinant of a square matrix. The matrix.determinant() method can also 
be used.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "3":
                            print("""
The inverse() method returns the inverse of a square matrix if it exists. This method uses the formula for the inverse matrix using the cofactor 
matrix and determinant.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "4":
                            print("""
The matrix.solve_system(vector) method takes a matrix and a vector and returns a vector containing the solution to the system of equations represented by the matrix and vector.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "5":
                            print("""
The matrix.transpose() method returns the transpose of the matrix by swapping the two indices.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "6":
                            print("""
The dunder method __pow__ is reprogrammed so that ** can be used to raise square matrices to the power of any integer. Raising a square matrix to the 
power of zero returns the identity matrix of the same size, raising a square matrix to the power of -1 returns the inverse matrix. This function does 
not use diagonalized matrices so it is slow for large powers, negative or positive.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "7":
                            print("""
The method matrix.exp() returns e raised to the power of the matrix. This is useful for linear differential equations. This method uses the Taylor 
series expansion of e to the power of a matrix up to the 20th term.
""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        if user_input == "q":
                            break
                        elif user_input == "b":
                            pass
                        else:
                            print("Did not understand user input")
                            continue
                        break
                    if user_input == "b":
                        continue
                    else:
                        break
                elif user_input == "3":
                    print("""
This module is incapable of finding eigenvalues and eigenvectors, and cannot diagonalize matrices. Thus, exponentiation is slower. Additionally, The 
determinant is also slow for large matrices, and because of how python handles complex numbers, may return a nonzero determinant when it is actually 
zero.""")
                    print("\nenter q to quit, or b to go back to the previous menu")
                    user_input = input("> ")
                    while (user_input != "q") and (user_input != "b"):
                        print("Did not understand user input")
                        user_input = input("> ")
                    if user_input == "b":
                        continue
                    elif user_input == "q":
                        break
                if user_input == "q":
                    break
                elif user_input == "b":
                    pass
                else:
                    print("Did not understand user input")
                    continue
                break
            if user_input == "b":
                pass
            else:
                break
        #Vectors
        elif user_input == "4":
            while True:
                print("Enter q anytime to quit, or b to go back")
                print("Enter a number:\n1: Defining a vector & representation\n2: Vector operations\n3: Note about covectors")
                user_input = input("> ")
                if user_input == "1":
                    print("""
Vectors in this module are a subclass of Tensors. Therefore, they inherit all the properties and methods of Tensors and have some vector specific 
functions. Their shape is of the form [n] where n is the number of rows and if n is negative the vector will be a covector. 
                          
Creating an instance of a vector is the same as creating any other tensor, except you can alternatively pass only a list and it will be made into a vector.
                          
When a vector is printed, the order, shape, and the list representing the vector are shown.""")
                    print("\nenter q to quit, or b to go back to the previous menu")
                    user_input = input("> ")
                    while (user_input != "q") and (user_input != "b"):
                        print("Did not understand user input")
                        user_input = input("> ")
                    if user_input == "b":
                        continue
                    elif user_input == "q":
                        break
                elif user_input == "2":
                    while True:
                        print("Enter q anytime to quit, or b to go back")
                        print("Enter a number:\n1: Scalar multiplication\n2: Dot product\n3: Cross Product\n4: Magnitude\n5: Angle between vectors\n6: Projection\n7: Components\n8: Linear independance")
                        user_input = input("> ")
                        if user_input == "1":
                            print("""
The dunder method __mul__ is reprogrammed so that * can be used to multiply vectors by scalars. Multiplication calls 
the Tensor class's tensor product method.  Order of multiplication does not matter here.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "2":
                            print("""
The method vector_1.dot_product(vector_2) returns the dot product of two vectors as a scalar. The dot product method calls tensor contraction from 
the Tensor class. * can be used to call __mul__ to multiply a covector by a conravariant vector, which is equivalent to the dot product. This only 
works if the index of the first vector is negatve for covariance and the index of the second vector is positive for contravariance.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "3":
                            print("""
The method vector_1.cross_product(vector_2) returns the cross product of two three dimensional vectors. The dot product is defined for seven 
dimensional vectors but that is not implimented in this module.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "4":
                            print("""
The dunder method __abs__ is reprogrammed so that abs(vector) can be used to find the magnitude of a vector. The method vector.magnitude() can also be used.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "5":
                            print("""
The method vector_1.angle_between(vector_2) will return the angle in radians between two vectors.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "6":
                            print("""
The method vector_1.projection(vector_2) returns the projection of vector_1 along vector_2 as a vector.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "7":
                            print("""
The method vector_1.component(vector_2) returns the component of vector_1 along vector_2 as a scalar.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        elif user_input == "8":
                            print("""
The static method module_alias.Vector() takes as many vectors as passed and returns True if they are linearly independant. This will return an error 
if under two vectors are passed.""")
                            print("\nenter q to quit, or b to go back to the previous menu")
                            user_input = input("> ")
                            while (user_input != "q") and (user_input != "b"):
                                print("Did not understand user input")
                                user_input = input("> ")
                            if user_input == "b":
                                continue
                            elif user_input == "q":
                                break
                        if user_input == "q":
                            break
                        elif user_input == "b":
                            pass
                        else:
                            print("Did not understand user input")
                            continue
                        break
                    if user_input == "b":
                        continue
                    else:
                        break
                elif user_input == "3":
                    print("""
Covectors are primarily used for the dot product and not much else. The dot product already exists with the vector_1.dot_product(vector_2) method, 
and gives no regard to whether the vectors are covariant or contravarient. Covectors do not have their own class, but if you wish to make one, simply 
make a vector with a negative index or make a vector and use the .change_index() method to change it to a covariant vector. If you wish to contract a 
covector with a contravariant vector, you can use the * operator between them, but keep in mind a contravariant vector multiplied by a covector will 
return an error. Covectors can still use the same methods as regular vectors.""")
                    print("\nenter q to quit, or b to go back to the previous menu")
                    user_input = input("> ")
                    while (user_input != "q") and (user_input != "b"):
                        print("Did not understand user input")
                        user_input = input("> ")
                    if user_input == "b":
                        continue
                    elif user_input == "q":
                        break
                if user_input == "q":
                    break
                elif user_input == "b":
                    pass
                else:
                    print("Did not understand user input")
                    continue
                break
            if user_input == "b":
                pass
            else:
                break
        #Quit
        elif user_input == "q":
            break
        elif user_input == "b":
            continue
        else:
            print("Did not understand user input")
    print("quit info")
