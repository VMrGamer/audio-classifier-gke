provider "google" {
  credentials = var.credentials_file
  project     = var.project_id
}

resource "google_storage_bucket" "terraform-backend" {
  name          = var.bucket_name
  location      = var.location

  force_destroy      = true
  bucket_policy_only = true

  versioning {
    enabled = true
  }
}