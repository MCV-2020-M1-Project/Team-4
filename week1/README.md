
# Deliverable Week 1

<!---Use this command to install needed dependencies:
<!---> pip install -r requirements.txt

### Execute QSD1
"""

<!---Usage:
  cbir.py <weekNumber> <teamNumber> <winEval> <querySet> <MethodNumber> [--testDir=<td>] 
  cbir.py -h | --help
  
<!---  <weekNumber> : Number of the week
       <teamNumber> : Team Number, in our case 04
       <winEval> : 0 for the first week
       <querySet> : number of the query
       <MethodNumber> : Number of the method : 1: Divided Histogram, 2: 3d Color Histogram
  
<!---  Example of use :  python cbir.py 1 04 0 1 1
       Options:
       --testDir=<td>        Directory with the test images & masks [default: /home/dlcv/DataSet/fake_test] 
  

### Execute QSD2
<!---Usage:
<!---  background_removal_results.py <weekNumber> <teamNumber> <winEval> <querySet> <MethodNumber> [--testDir=<td>] 
  background_removal_results.py -h | --help
  
 <!--- <weekNumber> : Number of the week
  <teamNumber> : Team Number, in our case 04
  <winEval> : 0 for the first week
  <querySet> : number of the query
  <MethodNumber> : Number of the method : 1: Edges, 2: Morph
  
  <!---Example of use : python background_removal_results.py 1 04 0 2 2
          
<!---Options:
  --testDir=<td>        Directory with the test images & masks [default: /home/dlcv/DataSet/fake_test]        ###Aixo del dir no ho tinc clar###
  


