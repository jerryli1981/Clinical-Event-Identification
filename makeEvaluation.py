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

    input_text_dir = os.path.join(plain_dir, "test")

    ann_dir = os.path.join(base_dir, 'annotation/coloncancer/Test')

    predict_span = []
    with open(os.path.join(base_dir, 'span_decision.txt') )as f:
        for l in f:
            predict_span.append(int(l.strip()))


    predict_type = []
    with open(os.path.join(base_dir, 'type_decisions.txt') )as f:
        for l in f:
            predict_type.append(int(l.strip()))

    predict_degree = []
    with open(os.path.join(base_dir, 'degree_decisions.txt') )as f:
        for l in f:
            predict_degree.append(int(l.strip()))

    predict_polarity = []
    with open(os.path.join(base_dir, 'polarity_decisions.txt') )as f:
        for l in f:
            predict_polarity.append(int(l.strip()))

    predict_modality = []
    with open(os.path.join(base_dir, 'modality_decisions.txt') )as f:
        for l in f:
            predict_modality.append(int(l.strip()))


    labelidx = 0
    
    for dir_path, dir_names, file_names in os.walk(input_text_dir):

        pbar = ProgressBar(maxval=len(file_names)).start()

        for i, fn in enumerate(sorted(file_names)):

            time.sleep(0.01)
            pbar.update(i + 1)

            # this for to make consistence
            for sub_dir, text_name, xml_names in anafora.walk(os.path.join(ann_dir, fn)):

                for xml_name in xml_names:

                    if "Temporal" not in xml_name:
                        continue

                    xml_path = os.path.join(ann_dir, text_name, xml_name)
                    data = anafora.AnaforaData.from_file(xml_path)

                    positive_span_label_map={}

                    for annotation in data.annotations:
                        if annotation.type == 'EVENT':

                            startoffset = annotation.spans[0][0]
                            endoffset = annotation.spans[0][1]

                            properties = annotation.properties
                            pros = {}
                            for pro_name in properties:
                                pro_val = properties.__getitem__(pro_name)
                                pros[pro_name] = pro_val
             
                            positive_span_label_map[(startoffset,endoffset)] = "1"

                    with open(os.path.join(input_text_dir, fn), 'r') as f:
                        content = f.read()

                    all_spans = content2span(content)

                    negative_span_label_map={}
                    for span in all_spans:
                        if span not in positive_span_label_map:
                            negative_span_label_map[span] = "0"


                    merged_spans = positive_span_label_map.keys() + negative_span_label_map.keys()


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

                        for span in merged_spans:
                            span_label = predict_span[labelidx]
                            type_label = predict_type[labelidx]
                            degree_label = predict_degree[labelidx]
                            polarity_label = predict_polarity[labelidx]
                            modality_label = predict_modality[labelidx]
                            labelidx += 1
                            if span_label == 1:
                                f.write("\t<entity>\n")
                                f.write("\t\t<id>"+str(count)+"@"+fn+"@system"+"</id>\n")
                                f.write("\t\t<span>"+str(span[0])+","+str(span[1])+"</span>\n")
                                f.write("\t\t<type>EVENT</type>\n")
                                f.write("\t\t<parentsType></parentsType>\n")
                                f.write("\t\t<properties>\n")
                                f.write("\t\t\t<DocTimeRel>BEFORE</DocTimeRel>\n")

                                if type_label == 1:
                                    f.write("\t\t\t<Type>"+"N/A"+"</Type>\n")
                                elif type_label == 2:
                                    f.write("\t\t\t<Type>"+"ASPECTUAL"+"</Type>\n")
                                elif type_label == 3:
                                    f.write("\t\t\t<Type>"+"EVIDENTIAL"+"</Type>\n")
                                else:
                                    f.write("\t\t\t<Type>"+"N/A"+"</Type>\n")

                                if degree_label == 1:
                                    f.write("\t\t\t<Degree>"+"N/A"+"</Degree>\n")
                                elif degree_label == 2:
                                    f.write("\t\t\t<Degree>"+"MOST"+"</Degree>\n")
                                elif degree_label == 3:
                                    f.write("\t\t\t<Degree>"+"LITTLE"+"</Degree>\n")
                                else:
                                    f.write("\t\t\t<Degree>"+"N/A"+"</Degree>\n")

                                if polarity_label == 1:
                                    f.write("\t\t\t<Polarity>"+"POS"+"</Polarity>\n")
                                elif polarity_label == 2:
                                    f.write("\t\t\t<Polarity>"+"NEG"+"</Polarity>\n")
                                else:
                                    f.write("\t\t\t<Polarity>"+"NEG"+"</Polarity>\n")

                                if modality_label == 1:
                                    f.write("\t\t\t<ContextualModality>"+"ACTUAL"+"</ContextualModality>\n")
                                elif modality_label == 2:
                                    f.write("\t\t\t<ContextualModality>"+"HYPOTHETICAL"+"</ContextualModality>\n")
                                elif modality_label == 3:
                                    f.write("\t\t\t<ContextualModality>"+"HEDGED"+"</ContextualModality>\n")
                                elif modality_label == 4:
                                    f.write("\t\t\t<ContextualModality>"+"GENERIC"+"</ContextualModality>\n")
                                else:
                                    f.write("\t\t\t<ContextualModality>"+"ACTUAL"+"</ContextualModality>\n")

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

