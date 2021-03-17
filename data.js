window.BENCHMARK_DATA = {
  "lastUpdate": 1615955070021,
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
      },
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
          "id": "e989d45073ae30d8c7ceedab9201b14135f981f5",
          "message": "Update README",
          "timestamp": "2021-03-17T15:20:31+11:00",
          "tree_id": "e0c919aa2cb5593633c01ef0450884724ffbb5dc",
          "url": "https://github.com/monash-emu/summer/commit/e989d45073ae30d8c7ceedab9201b14135f981f5"
        },
        "date": 1615955068226,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_default_ode_solver",
            "value": 0.07076200450630106,
            "unit": "iter/sec",
            "range": "stddev: 0.326258555584315",
            "extra": "mean: 14.131877792000003 sec\nrounds: 3"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_rk4_ode_solver",
            "value": 0.09757222348500606,
            "unit": "iter/sec",
            "range": "stddev: 0.0207594268240258",
            "extra": "mean: 10.248818406333337 sec\nrounds: 3"
          }
        ]
      }
    ]
  }
}