import os
from utils import content2span
from progressbar import ProgressBar
import time
from datetime import datetime

import anafora
if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.realpath(__file__))

    plain_dir = os.path.join(base_dir, 'original')

    output_dir = os.path.join(base_dir, 'output')

    input_text_test_dir = os.path.join(plain_dir, "test")

    ann_dir = os.path.join(base_dir, 'annotation/coloncancer/Test')

    predict = []
    with open(os.path.join(base_dir, 'decisions.txt') )as f:
        for l in f:
            predict.append(int(l.strip()))

    labelidx = 0
    

    for dir_path, dir_names, file_names in os.walk(input_text_test_dir):

        pbar = ProgressBar(maxval=len(file_names)).start()

        for i, fn in enumerate(file_names):

            time.sleep(0.01)
            pbar.update(i + 1)

            # this for to make consistence
            for sub_dir, text_name, xml_names in anafora.walk(os.path.join(ann_dir, fn)):

                for xml_name in xml_names:

                    with open(os.path.join(input_text_test_dir, fn), 'r') as f:
                        content = f.read()

                    spans = content2span(content)

                    dn = os.path.join(output_dir, fn)
                    if not os.path.exists(dn):
                        os.makedirs(dn)

                    outputAnn_path = os.path.join(dn, fn+"."+"Temporal-Relation.system.complete.xml")
                    with open(outputAnn_path, 'w') as f:
                        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n\n")
                        f.write("<data>\n")
                        f.write("<info>\n")
                        f.write("  <savetime>"+datetime.now().strftime('%H:%M:%S %d-%m-%Y')+"</savetime>\n")
                        f.write("  <progress>completed</progress>\n")
                        f.write("</info>"+"\n\n\n")
                        f.write("<schema path=\"./\" protocal=\"file\">temporal-schema.xml</schema>\n\n\n")
                        f.write("<annotations>\n\n\n")

                        count = 1
         
                        for span in spans:
                            label = predict[labelidx]
                            labelidx += 1

                            if label != 4:
                                f.write("\t<entity>\n")
                                f.write("\t\t<id>"+str(count)+"@"+fn+"@system"+"</id>\n")
                                f.write("\t\t<span>"+str(span[0])+","+str(span[1])+"</span>\n")
                                f.write("\t\t<type>EVENT</type>\n")
                                f.write("\t\t<parentsType></parentsType>\n")
                                f.write("\t\t<properties>\n")
                                f.write("\t\t\t<DocTimeRel>BEFORE</DocTimeRel>\n")
                                
                                if label == 1:
                                    f.write("\t\t\t<Type>"+"N/A"+"</Type>\n")
                                elif label == 2:
                                    f.write("\t\t\t<Type>"+"ASPECTUAL"+"</Type>\n")
                                elif label == 3:
                                    f.write("\t\t\t<Type>"+"EVIDENTIAL"+"</Type>\n")

                                f.write("\t\t\t<Degree>N/A</Degree>\n")
                                f.write("\t\t\t<Polarity>"+"POS"+"</Polarity>\n")
                                f.write("\t\t\t<ContextualModality>ACTUAL</ContextualModality>\n")
                                f.write("\t\t\t<ContextualAspect>N/A</ContextualAspect>\n")
                                f.write("\t\t\t<Permanence>UNDETERMINED</Permanence>\n")
                                f.write("\t\t</properties>\n")
                                f.write("\t</entity>\n\n")
                                count += 1
         
                        f.write("\n\n</annotations>\n")
                        f.write("</data>")

        pbar.finish()

        
    print "Total pred events is %d"%labelidx
    os.system("python -m anafora.evaluate -r annotation/coloncancer/Test/ -p output")

