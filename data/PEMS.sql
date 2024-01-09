create database traffic_flow;
CREATE TABLE `pems_5min` (
  `Timestamp` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `ID` int(11) DEFAULT NULL,
  `District` int(11) DEFAULT NULL,
  `Freeway` int(11) DEFAULT NULL,
  `Direction` char(10) DEFAULT NULL,
  `LaneType` char(10) DEFAULT NULL,
  `StationLength` float DEFAULT NULL,
  `numSamples` int(11) DEFAULT NULL,
  `percentObs` int(11) DEFAULT NULL,
  `TotalFlow` float DEFAULT NULL,
  `AvgOccupancy` float DEFAULT NULL,
  `AvgSpeed` float DEFAULT NULL,
  `LaneNSamples` int(11) DEFAULT NULL,
  `LaneNFlow` float DEFAULT NULL,
  `LaneNAvgOcc` float DEFAULT NULL,
  `LaneNAvgSpeed` float DEFAULT NULL,
  `LaneNObserved` int(11) DEFAULT NULL,
  `t1` float DEFAULT NULL,
  `t2` float DEFAULT NULL,
  `t3` float DEFAULT NULL,
  `t4` float DEFAULT NULL,
  `t5` float DEFAULT NULL,
  `t6` float DEFAULT NULL,
  `t7` float DEFAULT NULL,
  `t8` float DEFAULT NULL,
  `t9` float DEFAULT NULL,
  `t10` float DEFAULT NULL,
  `t11` float DEFAULT NULL,
  `t12` float DEFAULT NULL,
  `t13` float DEFAULT NULL,
  `t14` float DEFAULT NULL,
  `t15` float DEFAULT NULL,
  `t16` float DEFAULT NULL,
  `t17` float DEFAULT NULL,
  `t18` float DEFAULT NULL,
  `t19` float DEFAULT NULL,
  `t20` float DEFAULT NULL,
  `t21` float DEFAULT NULL,
  `t22` float DEFAULT NULL,
  `t23` float DEFAULT NULL,
  `t24` float DEFAULT NULL,
  `t25` float DEFAULT NULL,
  `t26` float DEFAULT NULL,
  `t27` float DEFAULT NULL,
  `t28` float DEFAULT NULL,
  `t29` float DEFAULT NULL,
  `t30` float DEFAULT NULL,
  `t31` float DEFAULT NULL,
  `t32` float DEFAULT NULL,
  `t33` float DEFAULT NULL,
  `t34` float DEFAULT NULL,
  `t35` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;