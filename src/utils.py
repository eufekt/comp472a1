def writeToFile(fileName,response):
    f = open("outputs/"+ fileName, "w")
    for item in response['arr']:
        f.write(str(item[0]).split('.')[0]+","+str(item[1]).split('.')[0]+"\n")
    f.write(str(response['report']))
    f.write(str(response['confusionMatrix']))
    f.close()