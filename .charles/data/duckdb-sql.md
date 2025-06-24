```sql

SELECT * FROM '.data/ylecun/mnist/train-00000-of-00001.parquet' LIMIT 5;

SELECT COUNT(*) FROM '.data/ylecun/mnist/train-00000-of-00001.parquet';

DESCRIBE '.data/ylecun/mnist/train-00000-of-00001.parquet';

SELECT * FROM '.data/ylecun/mnist/test-00000-of-00001.parquet' LIMIT 5;

SELECT COUNT(*) FROM '.data/ylecun/mnist/test-00000-of-00001.parquet';

DESCRIBE '.data/ylecun/mnist/test-00000-of-00001.parquet';

SELECT label  FROM '.data/ylecun/mnist/test-00000-of-00001.parquet' LIMIT 5;

-- ┌─────────────┬──────────────────────────────────┬─────────┬─────────┬─────────┬─────────┐
-- │ column_name │           column_type            │  null   │   key   │ default │  extra  │
-- │   varchar   │             varchar              │ varchar │ varchar │ varchar │ varchar │
-- ├─────────────┼──────────────────────────────────┼─────────┼─────────┼─────────┼─────────┤
-- │ image       │ STRUCT(bytes BLOB, path VARCHAR) │ YES     │ NULL    │ NULL    │ NULL    │
-- │ label       │ BIGINT                           │ YES     │ NULL    │ NULL    │ NULL    │
-- └─────────────┴──────────────────────────────────┴─────────┴─────────┴─────────┴─────────┘


-- Alternative approaches if len(image.bytes) fails:

-- 1. Try octet_length() function instead of len() (UPDATED)
SELECT 
    label,
    octet_length(image.bytes) as image_size_bytes
FROM '.data/ylecun/mnist/train-00000-of-00001.parquet' 
LIMIT 10;

-- 2. Check if bytes column is actually accessible
SELECT 
    label,
    image.path,
    typeof(image.bytes) as bytes_type
FROM '.data/ylecun/mnist/train-00000-of-00001.parquet' 
LIMIT 5;

-- 3. Try to cast or convert the bytes column
SELECT 
    label,
    length(cast(image.bytes as varchar)) as bytes_as_string_length
FROM '.data/ylecun/mnist/train-00000-of-00001.parquet' 
LIMIT 5;

-- 4. If bytes column is not accessible, use file path to estimate
SELECT 
    label,
    image.path,
    image.bytes IS NOT NULL as has_bytes_data
FROM '.data/ylecun/mnist/train-00000-of-00001.parquet' 
LIMIT 10;

-- 5. Alternative: Calculate estimated size based on MNIST dimensions (28x28 pixels)
-- Assuming grayscale images, each pixel = 1 byte
SELECT 
    label,
    CASE 
        WHEN image.bytes IS NOT NULL THEN 28 * 28 
        ELSE NULL 
    END as estimated_size_bytes
FROM '.data/ylecun/mnist/train-00000-of-00001.parquet' 
LIMIT 10;

-- Additional methods to get BLOB size when standard functions fail:



-- 7. Check if we can convert to base64 and measure that
SELECT 
    label,
    length(base64(image.bytes)) as base64_length,
    length(base64(image.bytes)) * 3 / 4 as estimated_bytes
FROM '.data/ylecun/mnist/train-00000-of-00001.parquet' 
LIMIT 5;


-- 11. If bytes are actually image pixels (28x28 for MNIST), estimate size
SELECT 
    label,
    CASE 
        WHEN image.bytes IS NOT NULL THEN 
            CASE 
                WHEN image.path LIKE '%.png' THEN 'PNG compressed'
                WHEN image.path LIKE '%.jpg' OR image.path LIKE '%.jpeg' THEN 'JPEG compressed'
                ELSE '784 bytes (28x28 grayscale)'
            END
        ELSE 'No image data'
    END as estimated_format_size
FROM '.data/ylecun/mnist/train-00000-of-00001.parquet' 
LIMIT 10;

```