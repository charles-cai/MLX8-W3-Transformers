```sql

SELECT * FROM '.data/ylecun/mnist/mnist/train-00000-of-00001.parquet' LIMIT 5;

SELECT COUNT(*) FROM '.data/ylecun/mnist/mnist/train-00000-of-00001.parquet';

DESCRIBE '.data/ylecun/mnist/mnist/train-00000-of-00001.parquet';

SELECT * FROM '.data/ylecun/mnist/mnist/test-00000-of-00001.parquet' LIMIT 5;

SELECT COUNT(*) FROM '.data/ylecun/mnist/mnist/test-00000-of-00001.parquet';

DESCRIBE '.data/ylecun/mnist/mnist/test-00000-of-00001.parquet';

SELECT label  FROM '.data/ylecun/mnist/mnist/test-00000-of-00001.parquet' LIMIT 5;

-- ┌─────────────┬──────────────────────────────────┬─────────┬─────────┬─────────┬─────────┐
-- │ column_name │           column_type            │  null   │   key   │ default │  extra  │
-- │   varchar   │             varchar              │ varchar │ varchar │ varchar │ varchar │
-- ├─────────────┼──────────────────────────────────┼─────────┼─────────┼─────────┼─────────┤
-- │ image       │ STRUCT(bytes BLOB, path VARCHAR) │ YES     │ NULL    │ NULL    │ NULL    │
-- │ label       │ BIGINT                           │ YES     │ NULL    │ NULL    │ NULL    │
-- └─────────────┴──────────────────────────────────┴─────────┴─────────┴─────────┴─────────┘
```