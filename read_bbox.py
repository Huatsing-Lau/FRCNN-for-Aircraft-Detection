from xml.etree.ElementTree import parse

#return labels and bboxes read from xml file
def bbox(xmlfile_name):
  tree = parse(xmlfile_name)
  root = tree.getroot()

  Results = root.find("Results")

  labels = []
  bboxes = []

  for Object in Results.findall("Object"):
    labels.append(Object.text)

  for Pixel in Results.findall("Pixel"):
    for Pt in Pixel:
      if eval(Pt.attrib["index"])==1:
        x1 = eval(Pt.attrib["LeftTopX"])
        y1 = eval(Pt.attrib["LeftTopY"])
      if eval(Pt.attrib["index"])==2:
        x2 = eval(Pt.attrib["LeftTopX"])
        y2 = eval(Pt.attrib["LeftTopY"])
      if eval(Pt.attrib["index"])==3:
        x3 = eval(Pt.attrib["LeftTopX"])
        y3 = eval(Pt.attrib["LeftTopY"])
      if eval(Pt.attrib["index"])==4:
        x4 = eval(Pt.attrib["LeftTopX"])
        y4 = eval(Pt.attrib["LeftTopY"])
    bboxes.append([(x1,y1), (x2,y2), (x3,y3), (x4,y4)])

  return labels, bboxes
