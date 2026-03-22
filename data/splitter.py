def assign_split(index: int, seed: int) -> str:
    """
    Deterministic split assignment based on index and seed.
    Approximate proportions:
      train        60%
      val           6%
      test_indist   2.4%
      test_ood      2.4%
      test_long     1.2%
    """
    import hashlib
    h = int(hashlib.md5(f"{seed}-{index}".encode()).hexdigest(), 16)
    r = h % 1000
    if r < 600:   return "train"
    if r < 660:   return "val"
    if r < 684:   return "test_indist"
    if r < 708:   return "test_ood"
    return "test_long"
