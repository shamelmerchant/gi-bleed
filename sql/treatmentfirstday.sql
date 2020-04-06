-- Original code:  https://github.com/cosgriffc/seq-severityscore
-- View for extraction treatment statuses on first day

DROP MATERIALIZED VIEW IF EXISTS treatmentfirstday CASCADE;
CREATE MATERIALIZED VIEW treatmentfirstday AS

SELECT trt.patientunitstayid
, MAX(CASE
	WHEN LOWER(trt.treatmentstring) LIKE '%vasopressors%' THEN 1
	ELSE 0
	END) AS vasopressor
, MAX(CASE
	WHEN LOWER(trt.treatmentstring) LIKE '%antiarrhythmics%' THEN 1
	ELSE 0
	END) AS antiarr
, MAX(CASE
	WHEN LOWER(trt.treatmentstring) LIKE '%antibacterials%' THEN 1
	ELSE 0
	END) AS abx
, MAX(CASE
	WHEN LOWER(trt.treatmentstring) LIKE '%sedative%' THEN 1
	ELSE 0
	END) AS sedative
, MAX(CASE
	WHEN LOWER(trt.treatmentstring) LIKE '%diuretic%' THEN 1
	ELSE 0
	END) AS diuretic
, MAX(CASE
	WHEN (LOWER(trt.treatmentstring) LIKE '%blood product%') AND (LOWER(trt.treatmentstring) LIKE '%packed red blood cells%') THEN 1
	ELSE 0
	END) AS blood_product_prbc
, MAX(CASE
		WHEN (LOWER(trt.treatmentstring) LIKE '%blood product%') AND (LOWER(trt.treatmentstring) NOT LIKE '%packed red blood cells%') THEN 1
		ELSE 0
		END) AS blood_product_other
, MAX(CASE
	WHEN LOWER(trt.treatmentstring) LIKE '%anti-inflammatories%' THEN 1
	ELSE 0
	END) AS antiinf
, MAX(CASE
		WHEN LOWER(trt.treatmentstring) LIKE '%antiplatelet%' THEN 1
		ELSE 0
		END) AS antiplatelet
, MAX(CASE
		WHEN (LOWER(trt.treatmentstring) LIKE '%anticoagulant%') AND ((LOWER(trt.treatmentstring) NOT LIKE '%heparin%') OR (LOWER(trt.treatmentstring) NOT LIKE '%enoxaparin%') ) THEN 1
		ELSE 0
		END) AS anticoagulant
FROM treatment trt
WHERE (trt.treatmentoffset <= 1440)
GROUP BY patientunitstayid;
