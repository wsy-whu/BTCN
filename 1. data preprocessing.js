//****************************0.数据准备**************************************
// 定义要筛选的县代码列表
// 导入FeatureCollection
var hubei_Wheat = ee.FeatureCollection("projects/ee-wangshuyuan/assets/Crop/Wheat/Wheat_test");
// 将codeID属性转为List
var codeIDList = hubei_Wheat.aggregate_array("county");//.filter(ee.Filter.gt("小麦面积(kha)", 15)).aggregate_array("县代码");
var codeIDList = ee.List(codeIDList).distinct();

var countyCodes = codeIDList;
var studyarea = ee.FeatureCollection("projects/ee-wangshuyuan/assets/SHP/SHP_County");	//县级矢量			FeatureCollection
// 使用ee.Filter.inList()来筛选包含在countyCodes列表中的县代码的特征
//var filteredCounties = studyarea.filter(ee.Filter.metadata("县代码","equals",110115));		//研究区矢量		FeatureCollection
var filteredCounties = studyarea.filter(ee.Filter.inList("县代码", countyCodes));
var Crop_Exist=ee.ImageCollection("projects/ee-wangshuyuan/assets/Crop/Wheat/CHN_Wheat_MA");//物候数据集（Planting Area）ImageCollection
var LST=ee.ImageCollection("MODIS/061/MOD11A1");
var VIs=ee.ImageCollection("MODIS/006/MOD13A2");			//世界范围时间序列EVI ImageCollection
var LST=LST.select("LST_Day_1km");
var VIs=VIs.select(["NDVI", "EVI"]);
// Daily mean 2m air temperature

//**************************************LST预处理阶段**********************************************
print("Hubei data:", hubei_Wheat);
/* print("Hubei data select:",hubei_Wheat.filter(ee.Filter.eq('年份', 2002))
													
													); */
///////////////////////////////////////////////////////////////
// 设置时间范围
var startDate = '2000-01-01';
var endDate = '2019-12-31';



// 获取起始年份和结束年份
var startYear = ee.Date(startDate).get('year');
var endYear = ee.Date(endDate).get('year');

// 定义一个月份列表
var months = ee.List.sequence(1, 12);



//**************************************VIs的月平均代码**********************************************
// 裁剪数据集，选择特定时间范围内的图像
VIs = VIs.filterDate(startDate, endDate);
var yearlyMonthlyMeans = ee.ImageCollection.fromImages(
  ee.List.sequence(startYear, endYear).map(function(year){
    return months.map(function(month){
      // 筛选出特定年份和月份的图像
      var filteredByYearMonth = VIs.filter(ee.Filter.calendarRange(year, year, 'year'))//关键在于改VIs
                                              .filter(ee.Filter.calendarRange(month, month, 'month'));
      
      // 计算月平均值
      var monthlyMean = filteredByYearMonth.mean();
      
      // 设置属性，表示年份和月份
      return monthlyMean.set({
        'year': year,
        'month': month
      });
    });
  }).flatten()
);

// 获取 ImageCollection 中的所有图像
var VI_list = yearlyMonthlyMeans.toList(yearlyMonthlyMeans.size());

// 获取除第一个图像之外的所有图像
var VIs= ee.ImageCollection.fromImages(
  ee.List.sequence(1, VI_list.length().subtract(1))
    .map(function(index) {
      return ee.Image(VI_list.get(index));
    })
);
//**************************************LST的月平均代码**********************************************
LST = LST.filterDate(startDate, endDate);
yearlyMonthlyMeans = ee.ImageCollection.fromImages(
  ee.List.sequence(startYear, endYear).map(function(year){
    return months.map(function(month){
      // 筛选出特定年份和月份的图像
      var filteredByYearMonth = LST.filter(ee.Filter.calendarRange(year, year, 'year'))//关键在于改VIs
                                              .filter(ee.Filter.calendarRange(month, month, 'month'));
      
      // 计算月平均值
      var monthlyMean = filteredByYearMonth.mean();
      
      // 设置属性，表示年份和月份
      return monthlyMean.set({
        'year': year,
        'month': month
      });
    });
  }).flatten()
);

// 获取 ImageCollection 中的所有图像
VI_list = yearlyMonthlyMeans.toList(yearlyMonthlyMeans.size());

// 获取除第一个图像之外的所有图像
var LST= ee.ImageCollection.fromImages(
  ee.List.sequence(1, VI_list.length().subtract(1))
    .map(function(index) {
      return ee.Image(VI_list.get(index));
    })
);
//**************************************Climate导入+月平均代码**********************************************
var Climates_mean = ee.ImageCollection('ECMWF/ERA5/MONTHLY')
                   //.select('mean_2m_air_temperature')
                   .filter(ee.Filter.date(startDate, endDate));
//*****************************************平均结束**************************************************
// 定义一个合并函数
function mergeBands(image1, image2Collection) {
  // 提取年份和月份
  var year = image1.get('year');
  var month = image1.get('month');

  // 在第二个ImageCollection中查找匹配的Image
  var matchedImage = image2Collection.filter(ee.Filter.equals('year', year))
                                     .filter(ee.Filter.equals('month', month))
                                     .first();

  // 如果找到了匹配的Image，则合并波段
  return ee.Image.cat(image1, matchedImage);
}

// 遍历一个ImageCollection，并应用合并函数
var All_bands = VIs.map(function(image) {
  return mergeBands(image, LST);
});
var All_bands = All_bands.map(function(image) {
  return mergeBands(image, Climates_mean);
});
print("Climates_mean: ", Climates_mean);
print("Climates_mean_filter1:", Climates_mean.filter(ee.Filter.equals('year', VIs.first().get('year')))
                                     .filter(ee.Filter.equals('month', 3))
                                     .first());
                                     print("Climates_mean: ", Climates_mean);
print("Climates_mean_filter2:", Climates_mean.filter(ee.Filter.equals('year', All_bands.first().get('year')))
                                     .filter(ee.Filter.equals('month', 3))
                                     .first());
Map.addLayer(LST.first(),{bands:['LST_Day_1km']},"LST");
Map.addLayer(VIs.first(),{bands:['EVI']},"VIs_EVI");
print("LST(mean)", LST);
print("VIs(mean):", VIs);
print("All_bands", All_bands);
print("filteredCounties:", filteredCounties);


// 打印结果
//var secondImage = ee.Image(yearlyMonthlyMeans.toList(yearlyMonthlyMeans.size()).get(1));

// 在地图上显示一个例子
//Map.addLayer(secondImage, {bands: ['NDVI'], min: 0, max: 9000, palette: ['FFFFFF', '00FF00']}, 'First Yearly Monthly Mean');
// print("filter exist:", Crop_Exist.filter(ee.Filter.eq('year', "2000")));
// print("first exist:", Crop_Exist.first());
///////////////////////////////0.批量年份mask/////////////////////////////////
// 定义提取年份信息的函数
var extractYear = function(image) {
  var index = image.getString("system:index");  // 获取 system:index 的值
  var year = index.split("_").get(3);  // 提取年份信息
  return image.set("year", ee.Number(year));  // 为图像设置 "year" 属性
};
// 使用 map() 函数生成 "year" 属性
var Crop_Exist = Crop_Exist.map(extractYear);
print("Crop_Exist(phenology):", Crop_Exist);
// 创建一个年份列表，从2000年到2015年
var years = ee.List.sequence(ee.Date(startDate).get('year'), ee.Date(endDate).get('year'));
//用于存储masked All_bands
var masked_All_bands = ee.ImageCollection([]);
// 遍历年份
years.getInfo().forEach(function(year) {
  // 从CropExist中选择相应年份的图像
  var maskImage = Crop_Exist
    .filter(ee.Filter.eq('system:index', 'CHN_Wheat_MA_'+year))
    .first();
  //print("maskImage",maskImage);
  // 从VIs中选择相应年份的图像
  var viImages = All_bands
    .filter(ee.Filter.eq('year', year));
  //print("viImages",viImages);
  // 如果存在相应年份的mask图像，则将VIs图像用mask图像进行遮罩
  var maskedVI = viImages.map(function(image) {
      return image.updateMask(maskImage);
  })
  masked_All_bands = masked_All_bands.merge(maskedVI);
    // 在这里，你可以使用maskedVI进行进一步的处理或导出
    // 例如，将遮罩后的图像添加到一个新的图像集合中
    // 这里只是简单的示例，你可以根据需要进行修改
    //print("Masked VI for year " + year, maskedVI);

});
// 打印有效masked VIs(ImageCollection)时空信息都是多段
//masked_All_bands = All_bands;
Map.centerObject(filteredCounties);
print("物候区域mask的All_bands数据:", masked_All_bands);
Map.addLayer(masked_All_bands.first(),{bands:['EVI']},"masked_All_bands");
Map.addLayer(filteredCounties);
////////////////////////////////////////////////////////////////////
//*****************1.基于Phenological，对研究区进行筛选********************
// 选择要处理的VI数据和Area数据：clipEVI	clipArea
// 使用Area图层更新VI数据的掩码

/* var masked_All_bands = All_bands.map(function(image) {
  // 使用Area图层更新掩码
  var mask = Crop_Exist.filter(ee.Filter.eq('year', image.get('year')));
  var updatedMask = image.updateMask(mask);
  return updatedMask;
}); */

//FeatureCollection:  studyarea     filteredCounties
//ImageCollection:		Crop_Exist(物候clipArea)	EVI(全球clipEVI)
//ImageCollection:		物候clipArea裁剪clipEVI：masked_All_bands	

//*******************3.数据筛选完成后，对区域求平均、累加并导出，对研究区进行筛选********************
//**************************有个函数叫reduceRegions，批量求的时候可以考虑这个************************
//****************************根据产量数据集整理reduceRegions的排列**********************************
//masked_All_bands经过County和Phenology筛选后仍然是多时序多波段
//定义一个函数，用于筛选有效数据并计算空间尺度上的平均值
// var sumFeature = ee.FeatureCollection([]);
//************************************************最终直方图整合****************************************
var VIs_NDVI_hist_min = 0;   //statistics.get('NDVI_min');
var VIs_NDVI_hist_max = 8000 //statistics.get('NDVI_max');
var VIs_EVI_hist_min = 0;   //statistics.get('NDVI_min');
var VIs_EVI_hist_max = 5000 //statistics.get('NDVI_max');
var LST_Day_hist_min = 0;
var LST_Day_hist_max = 30;
var Temp_hist_min = 273-8;
var Temp_hist_max = 273+30;
var Pre_hist_min = 0;
var Pre_hist_max = 0.05;
var Pressure_surface_hist_min = 102000;
var Pressure_surface_hist_max = 103000;
var Pressure_sea_hist_min = 102000;
var Pressure_sea_hist_max = 103000;
var Wind_hist_min = -1;
var Wind_hist_max = 2;

//var hist_buckets = 20;
var hist_scale = 1000;		//Resolution



// 定义函数，用于计算直方图
var computeHistogram = function(image, geometry, scale, hist_min, hist_max, band) {
  // 定义直方图参数
  var hist_buckets = 20;

  // 使用 reduceRegion 函数计算范围min-max
  var hist_range = image.reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: geometry,
    scale: scale,
    tileScale:8,
    maxPixels: 1e9
  });
  // 使用 reduceRegion 函数计算直方图
  var histogram = image.reduceRegion({
    reducer: ee.Reducer.fixedHistogram(hist_range.get(band + '_min'), hist_range.get(band + '_max'), hist_buckets),
    geometry: geometry,
    scale: scale,
    tileScale:8,
    maxPixels: 1e9
  });

  return histogram;
};

/* var temp = masked_All_bands.first();
print("temp_first():",temp);
print("temp_year",temp.get("year"));
print("yield", hubei_Wheat.filter(ee.Filter.eq('县代码', filteredCounties.first().get("县代码")))
                          .first().get("小麦单产(kg/ha)")); */
                          //.filter(ee.Filter.eq('﻿年份', 2000)));
												//	.first().get("小麦单产(kg/ha)"));
//print("histogram:", computeHistogram(secondImage, filteredCounties, hist_scale, VIshist_min, VIshist_max));
var Hist_List = function(image) {
  // 使用updateMask函数筛选有效数据
  //var threshold = 0; // 根据你的数据特性设定阈值
	/* var FeatureMean = function(Feature) {
		var meanDictionary = image.reduceRegion({
			reducer: ee.Reducer.mean(),
			geometry: Feature.geometry(),
			scale: 1000 // 设定分辨率，根据你的数据特性设定
		});
	var meanValue = meanDictionary.getNumber("EVI"); // 将"your_band_name"替换为你要计算平均值的波段名称 
	return ee.Feature(null, {"date": image.id(), "county": ee.Number(Feature.get("县代码")), "mean_value": meanValue});
	}*/
	
	var FeatureMean = function(Feature) {
		return ee.Feature(null, {"year": image.get("year"), 
								"month": image.get("month"), 
								"county": ee.Number(Feature.get("县代码")), 
								//"codeID": (ee.Number(image.get("year"))-2000)*1000000+ee.Number(Feature.get("县代码")),
								// "yield": hubei_Wheat.filter(ee.Filter.eq('﻿年份', ee.Number(image.get("year"))))
													// .filter(ee.Filter.eq('县代码', ee.Number(Feature.get("县代码"))))
													// .get("小麦单产(kg/ha)"),
								"NDVI_Hist": computeHistogram(image.select("NDVI"), Feature.geometry(), hist_scale, VIs_NDVI_hist_min, VIs_NDVI_hist_max, "NDVI"),
								"EVI_Hist": computeHistogram(image.select("EVI"), Feature.geometry(), hist_scale, VIs_EVI_hist_min, VIs_EVI_hist_max, "EVI"),
								
								"mean_2m_air_temperature": computeHistogram(image.select("mean_2m_air_temperature"), Feature.geometry(), hist_scale, Temp_hist_min, Temp_hist_max, "mean_2m_air_temperature"),
								"minimum_2m_air_temperature": computeHistogram(image.select("minimum_2m_air_temperature"), Feature.geometry(), hist_scale, Temp_hist_min, Temp_hist_max, "minimum_2m_air_temperature"),
								"maximum_2m_air_temperature": computeHistogram(image.select("maximum_2m_air_temperature"), Feature.geometry(), hist_scale, Temp_hist_min, Temp_hist_max, "maximum_2m_air_temperature"),
								"dewpoint_2m_temperature": computeHistogram(image.select("dewpoint_2m_temperature"), Feature.geometry(), hist_scale, Temp_hist_min, Temp_hist_max, "dewpoint_2m_temperature"),
								
								"total_precipitation": computeHistogram(image.select("total_precipitation"), Feature.geometry(), hist_scale, Pre_hist_min, Pre_hist_max, "total_precipitation"),
								
								"surface_pressure": computeHistogram(image.select("surface_pressure"), Feature.geometry(), hist_scale, Pressure_surface_hist_min, Pressure_surface_hist_max, "surface_pressure"),
								
								"mean_sea_level_pressure": computeHistogram(image.select("mean_sea_level_pressure"), Feature.geometry(), hist_scale, Pressure_sea_hist_min, Pressure_sea_hist_max, "mean_sea_level_pressure"),
								
								"u_component_of_wind_10m": computeHistogram(image.select("u_component_of_wind_10m"), Feature.geometry(), hist_scale, Wind_hist_min, Wind_hist_max, "u_component_of_wind_10m"),
								"v_component_of_wind_10m": computeHistogram(image.select("v_component_of_wind_10m"), Feature.geometry(), hist_scale, Wind_hist_min, Wind_hist_max, "v_component_of_wind_10m"),
								});//meanValue
	}
  // 使用reduceRegion函数计算每个图像中有效像素的平均值
	return filteredCounties.map(FeatureMean);	//ee.Feature(null, {"mean_value": meanValue});
};
var Hist_All_bands = masked_All_bands.map(Hist_List);
var Hist_All_bands_flatten = Hist_All_bands.flatten();
print("Hist_All_bands_flatten:", Hist_All_bands_flatten);
								//.filter(ee.Filter.eq('year', 2002))
								//.filter(ee.Filter.eq('month', 2)));

var exportOptions = {
  collection: Hist_All_bands_flatten,
  description: 'Hist_All_bands_free', // 导出文件的名称
  fileFormat: 'csv' // 导出文件的格式，可以选择其他格式如 GeoJSON
};

// 启动导出任务
Export.table.toDrive(exportOptions);

/* var histogram = secondImage.reduceRegion({
  reducer: ee.Reducer.fixedHistogram(hist_min, hist_max, hist_buckets),
  geometry: filteredCounties,
  scale: 1000, // 根据需要设置适当的比例尺
  maxPixels: 1e9 // 根据需要设置最大像素数
}); */


//print("物候+县筛选平均的EVI:", meanEVI1);

//Map.addLayer(meanVI.first(),clipEVI_show,"mean EVI");//masked_All_bands筛选后仍然是多时序多波段，first为了时序，clipshow为了波段
/* // 时间尺度求平均，保留多波段信息，但是将Colletion代表的时间信息求平局生成了image。如果要选时间段就先filter再平均
var meanVICollection = masked_All_bands.reduce(ee.Reducer.mean());// 获取平均值
//var meanEVI = meanVICollection.getNumber("EVI"); //获取波段 
print("meanVICollection:", meanVICollection);
Map.addLayer(meanVICollection); */

//**************************4.将批量处理好的数据导出**********************************
//最后的流程代码很简单：先按时间求平均，再将imageCollection用reduceRegions裁剪多个区域平均（县代码怎么分的还要考虑一下）导出
//还是要考虑Phenological的问题，是不是应该全图直接mask一下






















//*****************备用参考代码*********************
//Map.centerObject(filteredCounties);  //锚定中心 数字为缩放尺度 1
//Map.addLayer(filteredCounties); //展示所选县的范围  1
//print("filteredCounties 的属性信息：", filteredCounties);
//print(ee.Filter.eq('县代码', '152531'))
// 在地图上显示有效EVI数据
// Map.addLayer(masked_All_bands, {
  // min: 0,
  // max: 1,
  // palette: ['blue', 'green', 'red']
// }, '有效的VI数据');

//区域平均：
/* var areaMean = function(image) {
  // 使用updateMask函数筛选有效数据
  //image = image.updateMask(image.gte(threshold));
  
  // 计算有效区域的平均值
  var meanCollection = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    //geometry: regionOfInterest,
    scale: 1000 // 设定分辨率，根据你的数据特性设定
  });
  
  // 获取平均值，即从meanDictionary这个Collection里面选取所需波段(单个image可包含多个波段)
  var meanValue = meanCollection.getNumber("EVI"); // 将"your_band_name"替换为你要计算平均值的波段名称
  
  // 创建一个新的属性，将平均值作为属性值
  return image.set("mean_value", meanValue);
};

// 使用map函数将calculateMean函数应用于imageCollection中的每个影像（单个image可包含多个波段）
//	calculateMean平均出来的ImageCollection是时间序列，而其中每个image包含的是多个波段，已经在平均函数里剔出特定波段了 */


//*******************1.用FeatureCollection裁剪数据区域(缺一个批量县裁剪处理）**********************************
//先定义一个对单幅影像裁剪的function
/* // function clipImg(image){
  // return image.clipToCollection(filteredCounties);//其中fc是行政区矢量文件
// }
//对ImageCollection运行这个function加以裁剪
var clipEVI=EVI.map(clipImg);
var clipArea=Crop_Exist.map(clipImg);
//筛选波段并显示
print(clipEVI);//将信息打印在console上看时空分辨率
print(clipArea);
Map.centerObject(filteredCounties);  //锚定中心 数字为缩放尺度 1
var clipEVI_show={bands:['EVI']};
var clipArea_show={bands:['b1']};
Map.addLayer(clipEVI,clipEVI_show,"clipEVI");
Map.addLayer(clipArea,clipArea_show,"clipArea"); */