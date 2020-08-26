import xlrd
import numpy as np
from Preprocessing import pre_process_text
from Extraction import vectorize_test, build_model,cont_vect1,cont_vect2


# get relevant informarion from a given file
def Extract_preprocess(filename, sheet_name) :
    # open the excel file
    wb1 = xlrd.open_workbook(filename)

    # get the data sheet
    sh1 = wb1.sheet_by_name(sheet_name)

    vector= np.array([])
    for i in range(sh1.nrows):
        vector=[]
        if i > 0:

            #name
            name=sh1.row_values(i)[2]
            name=pre_process_text(name)
            vector.append(name)

            #description
            description=sh1.row_values(i)[4]
            description=pre_process_text(description)
            vector.append(description)

            yield vector


# main function
def main():

    # get model
    model=build_model()


    name_array=[]
    description_array=[]

    for vector in Extract_preprocess('prodis_product_extract.xlsx', 'Sheet1'):
        name_array.append(vector[0])
        description_array.append(vector[1])

    # vectorize
    names=vectorize_test(name_array, "name")
    descriptions=vectorize_test(description_array, "description")


    k=0

    for name,description in zip(names, descriptions):

        vector=name.tolist()
        vector.extend(description.tolist())


        if k==0:
            vectors= np.array(np.array(vector))
            k=1
        else :
            vectors=np.vstack((vectors, np.array(vector)))

    i=2
    with open('results','w') as f :
        f.write ('Row'+ '\t'+ 'Category'+ '\n')
        for result in model.predict(vectors):
            f.write (str(i)+ '\t'+ str(result)+ '\n')
            i+=1
    f.close()









if __name__ == '__main__':

    main()
