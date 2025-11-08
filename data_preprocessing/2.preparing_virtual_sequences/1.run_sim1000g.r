#!/usr/bin/env Rscript
library(sim1000G)
library(data.table)

simulate_and_save_vcf <- function(filename, numeric_chrom,
                                  input_dir = "/path/to/0.vcf_preprocessing.sh/output_dir",
                                  output_dir = "/path/to/output_dir", seed = 42) {
  if (!is.null(seed)) set.seed(seed)
  input_vcf_path  <- file.path(input_dir, filename)
  gene_id         <- sub("\\.training_samples\\.vcf\\.gz$", "", filename)
  output_filename <- paste0(gene_id, "_simulated_chr", numeric_chrom, ".vcf")
  output_vcf_path <- file.path(output_dir, output_filename)

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  cat("▶ start simulation:", filename, "\n")

  vcf <- readVCF(input_vcf_path, maxNumberOfVariants = Inf, min_maf = 0, max_maf = 1.0)

  vcf$vcf$CHROM <- rep(as.character(numeric_chrom), nrow(vcf$vcf))

  readGeneticMap(numeric_chrom)

  startSimulation(vcf, totalNumberOfIndividuals = 1000)
  generateUnrelatedIndividuals(1000)

  save_simulation_to_vcf(vcf, output_vcf_path)
}

save_simulation_to_vcf <- function(vcf, output_vcf_path, sample_ids = NULL) {
  if (is.null(sample_ids)) {
    sample_ids <- paste0("sample", 1:SIM$individuals_generated)
  }

  chrom <- unique(vcf$vcf$CHROM)
  pos   <- vcf$vcf$POS
  ref   <- vcf$vcf$REF
  alt   <- vcf$vcf$ALT

  n_ind <- SIM$individuals_generated
  n_var <- ncol(SIM$gt1)

  vcf_header <- c(
    "##fileformat=VCFv4.2",
    "##FILTER=<ID=PASS,Description=\"All filters passed\">",
    "##source=sim1000G",
    paste0("##contig=<ID=", chrom, ">"),
    "##INFO=<ID=.,Number=.,Type=String,Description=\".\">",
    "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
    "##bcftools_viewVersion=1.6+htslib-1.6",
    paste0("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t",
           paste(sample_ids, collapse = "\t"))
  )


  gt_mat <- matrix(nrow = n_var, ncol = n_ind)
  for (i in seq_len(n_ind)) {
    g1 <- SIM$gt1[i, ]
    g2 <- SIM$gt2[i, ]
    gt_mat[, i] <- paste0(g1, "|", g2)
  }

  vcf_body <- vapply(seq_len(n_var), function(i) {
    paste(
      chrom, pos[i], paste0("var", i), ref[i], alt[i],
      ".", ".", ".", "GT",
      paste(gt_mat[i, ], collapse = "\t"),
      sep = "\t"
    )
  }, FUN.VALUE = character(1))

  writeLines(c(vcf_header, vcf_body), con = output_vcf_path)
  message(paste("✅ VCF save complete:", output_vcf_path))
}

run_batch_simulation <- function(csv_path = "/data/9.vcf_chr.csv") {
  df <- fread(csv_path)

  for (i in seq_len(nrow(df))) {
    tryCatch({
      simulate_and_save_vcf(
        filename      = df$FileName[i],
        numeric_chrom = df$NumericChrom[i],
        seed = 42
      )
    }, error = function(e) {
      message(paste("❌ error occurred -", df$FileName[i], ":", e$message))
    })
  }
}

run_batch_simulation()