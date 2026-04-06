# MediaWiki Docker Setup - Troubleshooting Notes

**Date:** 2026-03-25  
**Task:** Copy LocalSettings.php from web installer to Docker container

---

## The Process

### 1. Web Installer Setup
- Complete MediaWiki web setup at `http://your-server/wiki`
- Web installer asks you to download `LocalSettings.php`
- File is saved to your Downloads folder (or current directory)

### 2. Copy to Container
```bash
docker cp LocalSettings.php <container_name>:/var/www/html/
```

---

## Glitches Encountered

### Glitch 1: "no such directory" Error
**Symptom:**
```
$ docker cp LocalSettings.php mediawiki:/var/www/html/wiki/
no such directory
```

**Cause:** The path `/var/www/html/wiki/` doesn't exist inside the container. MediaWiki is installed directly in `/var/www/html/`.

**Fix:**
```bash
docker cp LocalSettings.php mediawiki:/var/www/html/
```

---

### Glitch 2: Tab Completion Doesn't Work for Destination
**Symptom:**
- `docker cp LocalSettings.php mediawiki:/var/www/html/<tab>` - no completions

**Cause:** Expected behavior. Your local shell has no visibility into the container's filesystem, so it can't offer tab completions for container paths.

**Note:** Tab completion works for the **source** file (your local machine) but not the **destination** (inside container).

---

### Glitch 3: "closed pipe" Error
**Symptom:**
```
time="2026-03-25T06:18:03Z" level=error msg="Can't add file /home/miles/LocalSettings.php to tar: io: read/write on closed pipe"
```

**Cause:** Side-effect of the destination path not existing. Once you fix the path, this error goes away.

---

## Debugging Commands

### Find actual directory structure in container:
```bash
docker exec mediawiki ls -la /var/www/html/
```

### Verify source file exists:
```bash
ls -la ~/LocalSettings.php
```

### List running containers:
```bash
docker ps
```

---

## Correct Command (Final)

```bash
docker cp LocalSettings.php mediawiki:/var/www/html/
```

---

## After Copying

Optionally restart the container:
```bash
docker restart mediawiki
```

Then refresh your wiki page and it should be fully functional!
