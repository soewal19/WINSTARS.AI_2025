from PIL import Image
import xml.etree.ElementTree as ET
import os

def create_voc_xml(img_path, bboxes, labels, out_xml):
    img = Image.open(img_path)
    W,H = img.size
    ann = ET.Element('annotation')
    ET.SubElement(ann,'folder').text = os.path.basename(os.path.dirname(img_path))
    ET.SubElement(ann,'filename').text = os.path.basename(img_path)
    size = ET.SubElement(ann,'size')
    ET.SubElement(size,'width').text=str(W)
    ET.SubElement(size,'height').text=str(H)
    ET.SubElement(size,'depth').text='3'
    for bbox,label in zip(bboxes,labels):
        obj = ET.SubElement(ann,'object')
        ET.SubElement(obj,'name').text = label
        bnd = ET.SubElement(obj,'bndbox')
        xmin,ymin,xmax,ymax = bbox
        ET.SubElement(bnd,'xmin').text=str(int(xmin))
        ET.SubElement(bnd,'ymin').text=str(int(ymin))
        ET.SubElement(bnd,'xmax').text=str(int(xmax))
        ET.SubElement(bnd,'ymax').text=str(int(ymax))
    tree = ET.ElementTree(ann)
    os.makedirs(os.path.dirname(out_xml), exist_ok=True)
    tree.write(out_xml)
