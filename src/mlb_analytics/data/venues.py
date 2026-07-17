# Approximate park coordinates and neutralized park factors. Update annually from your preferred licensed source.
VENUES={
'Chase Field':(33.4455,-112.0667,1.01),'Truist Park':(33.8907,-84.4677,1.02),'Oriole Park at Camden Yards':(39.2839,-76.6217,.98),
'Fenway Park':(42.3467,-71.0972,1.04),'Wrigley Field':(41.9484,-87.6553,1.03),'Rate Field':(41.8300,-87.6338,1.01),
'Great American Ball Park':(39.0979,-84.5082,1.07),'Progressive Field':(41.4962,-81.6852,.99),'Coors Field':(39.7559,-104.9942,1.18),
'Comerica Park':(42.3390,-83.0485,.96),'Daikin Park':(29.7573,-95.3555,1.01),'Kauffman Stadium':(39.0517,-94.4803,.96),
'Angel Stadium':(33.8003,-117.8827,.98),'Dodger Stadium':(34.0739,-118.2400,1.00),'loanDepot park':(25.7781,-80.2197,.96),
'American Family Field':(43.0280,-87.9712,1.02),'Target Field':(44.9817,-93.2776,.99),'Citi Field':(40.7571,-73.8458,.97),
'Yankee Stadium':(40.8296,-73.9262,1.05),'Sutter Health Park':(38.5803,-121.5130,1.00),'Citizens Bank Park':(39.9061,-75.1665,1.04),
'PNC Park':(40.4469,-80.0057,.97),'Petco Park':(32.7076,-117.1570,.95),'Oracle Park':(37.7786,-122.3893,.94),
'T-Mobile Park':(47.5914,-122.3325,.95),'Busch Stadium':(38.6226,-90.1928,.98),'George M. Steinbrenner Field':(27.9800,-82.5066,1.00),
'Globe Life Field':(32.7473,-97.0847,1.01),'Rogers Centre':(43.6414,-79.3894,1.03),'Nationals Park':(38.8730,-77.0074,1.00)}
def venue_info(name): return VENUES.get(name,(None,None,1.0))
