const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const ICON_SIZES = [16, 32, 72, 96, 128, 144, 152, 167, 180, 192, 384, 512];

const SOURCE_SVG = path.join(__dirname, "../public/logo.svg");
const OUTPUT_DIR = path.join(__dirname, "../public/icons");

// Create output directory if it doesn't exist
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Generate PNG icons for each size
ICON_SIZES.forEach((size) => {
  sharp(SOURCE_SVG)
    .resize(size, size)
    .png()
    .toFile(path.join(OUTPUT_DIR, `icon-${size}x${size}.png`))
    .then(() => {
      console.log(`Generated ${size}x${size} icon`);
    })
    .catch((err) => {
      console.error(`Error generating ${size}x${size} icon:`, err);
    });
});

// Generate special icons
const SPECIAL_ICONS = [
  { name: "apple-icon.png", size: 180 },
  { name: "favicon.png", size: 32 },
  { name: "safari-pinned-tab.svg", size: 512 },
];

SPECIAL_ICONS.forEach((icon) => {
  const output = path.join(OUTPUT_DIR, icon.name);

  if (icon.name.endsWith(".svg")) {
    fs.copyFileSync(SOURCE_SVG, output);
    console.log(`Copied ${icon.name}`);
  } else {
    sharp(SOURCE_SVG)
      .resize(icon.size, icon.size)
      .toFormat(icon.name.endsWith(".ico") ? "ico" : "png")
      .toFile(output)
      .then(() => {
        console.log(`Generated ${icon.name}`);
      })
      .catch((err) => {
        console.error(`Error generating ${icon.name}:`, err);
      });
  }
});
