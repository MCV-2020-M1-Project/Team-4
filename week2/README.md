
# Deliverable Week 2

Use this command to install needed dependencies:
pip install -r requirements.txt

### Execute QSD1

Usage:
  cbir.py |weekNumber| |teamNumber| |winEval| |querySet| |MethodNumber| |distanceMeasure|
  cbir.py -h | --help
  
  weekNumber --> Number of the week

  teamNumber --> Team Number, in our case 04
  
  winEval --> 0 for the first week and 1 for the rest of weeks
  
  querySet --> number of the query
  
  MethodNumber --> Number of the method : 1: Divided Histogram, 2: 3d Color Histogram
  
  distanceMeasure --> 1: Euclidean distance, 2: x^2 distance
  
  ### Example of use --> python cbir.py 1 04 0 1 1 2


### Execute QSD2

Usage:
  background_removal_results.py |weekNumber| |teamNumber| |winEval| |querySet| |MethodNumber| |distanceMeasure| 
  background_removal_results.py -h | --help
  
  weekNumber --> Number of the week
  
  teamNumber --> Team Number, in our case 04
  
  winEval --> 0 for the first week and 1 for the rest of weeks
  
  querySet --> number of the query
  
  MethodNumber --> Number of the method : 1: Edges, 2: Morph
  
  distanceMeasure --> 1: Euclidean distance, 2: x^2 distance
  
  ### Example of use --> python background_removal_results.py 1 04 0 2 2 1




