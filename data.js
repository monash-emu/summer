window.BENCHMARK_DATA = {
  "lastUpdate": 1615954714921,
  "repoUrl": "https://github.com/monash-emu/summer",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "mattdsegal@gmail.com",
            "name": "Matthew Segal",
            "username": "MattSegal"
          },
          "committer": {
            "email": "mattdsegal@gmail.com",
            "name": "Matthew Segal",
            "username": "MattSegal"
          },
          "distinct": true,
          "id": "9745ad55d0a067febc1da7450f58245589b14a64",
          "message": "Remove benchmark JSON",
          "timestamp": "2021-03-17T15:07:31+11:00",
          "tree_id": "ab24f8eedd4013cb2ab155d5e13b5cdb14a0a222",
          "url": "https://github.com/monash-emu/summer/commit/9745ad55d0a067febc1da7450f58245589b14a64"
        },
        "date": 1615954713106,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_default_ode_solver",
            "value": 0.07885548258046124,
            "unit": "iter/sec",
            "range": "stddev: 0.09375282600623916",
            "extra": "mean: 12.681426418000001 sec\nrounds: 3"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_rk4_ode_solver",
            "value": 0.10368733006567483,
            "unit": "iter/sec",
            "range": "stddev: 0.013364314413347574",
            "extra": "mean: 9.644379880999992 sec\nrounds: 3"
          }
        ]
      }
    ]
  }
}