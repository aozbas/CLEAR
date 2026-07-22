import { readFile, readdir } from "node:fs/promises";
import { relative } from "node:path";
import { fileURLToPath } from "node:url";

const mobileRoot = new URL("../", import.meta.url);
const sourceRoot = new URL("../src/", import.meta.url);
const forbiddenFiles = [
  "src/lib/hosted_database.ts",
  "src/screens/LoginScreen.tsx",
  "src/screens/ScanScreen.tsx",
  "src/screens/HistoryScreen.tsx",
];
const forbiddenPatterns = [
  /@hosted_database\//i,
  /hosted_database\.auth/i,
  /\/scans(?:["'`/]|$)/i,
  /\/predictions["'`]/i,
  /signInWithPassword/i,
  /signUp\s*\(/i,
  /saved to history/i,
];

const sourceFiles = [
  new URL("../App.tsx", import.meta.url),
  ...(await collectSourceFiles(sourceRoot)),
];
const violations = [];

for (const file of sourceFiles) {
  const text = await readFile(file, "utf8");
  for (const pattern of forbiddenPatterns) {
    if (pattern.test(text)) {
      violations.push(
        `${relative(fileURLToPath(mobileRoot), fileURLToPath(file))} matches ${pattern}`,
      );
    }
  }
}

const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const packageLock = await readFile(new URL("../package-lock.json", import.meta.url), "utf8");
const apiSource = await readFile(new URL("../src/lib/api.ts", import.meta.url), "utf8");
const demoSource = await readFile(
  new URL("../src/screens/DemoScanScreen.tsx", import.meta.url),
  "utf8",
);
const labelSource = await readFile(new URL("../src/lib/labels.ts", import.meta.url), "utf8");
const appConfig = await readFile(new URL("../app.json", import.meta.url), "utf8");
if (packageJson.dependencies?.["@hosted_database/hosted_database-js"]) {
  violations.push("package.json still depends on @hosted_database/hosted_database-js");
}
if (packageLock.includes("@hosted_database/hosted_database-js")) {
  violations.push("package-lock.json still contains @hosted_database/hosted_database-js");
}

const requiredApiControls = [
  "FileSystemUploadType.BINARY_CONTENT",
  "REQUEST_TIMEOUT_MS",
  "MAX_IMAGE_BYTES",
  "removeTemporaryPickerFile",
  "errorMessageForStatus",
  "ensureSecureApiBase",
  '"classification_available"',
  '"classifier_uncertain"',
  '"poor_image_quality"',
  '"unsupported_image"',
];
for (const control of requiredApiControls) {
  if (!apiSource.includes(control)) violations.push(`src/lib/api.ts is missing ${control}`);
}
if (apiSource.includes("response.text()")) {
  violations.push("src/lib/api.ts exposes an upstream response body");
}
if (!apiSource.includes('relativePath.split(/[\\\\/]/).includes("..")')) {
  violations.push("src/lib/api.ts is missing the cache traversal guard");
}
if (
  !/controller\.signal\.aborted\s*\|\|\s*requestController\.current\s*!==\s*controller/.test(
    demoSource,
  )
) {
  violations.push("DemoScanScreen.tsx can publish a superseded upload result");
}
if (!demoSource.includes("No classification shown")) {
  violations.push("DemoScanScreen.tsx does not render an explicit abstention state");
}
if (appConfig.includes("share them with your friends")) {
  violations.push("app.json still uses the default image-picker permission copy");
}

const requiredModelLabels = [
  "actinic_keratosis",
  "basal_cell_carcinoma",
  "melanoma",
  "nevus",
  "squamous_cell_carcinoma",
  "seborrheic_keratosis",
];
for (const label of requiredModelLabels) {
  if (!labelSource.includes(label)) violations.push(`src/lib/labels.ts is missing ${label}`);
}
for (const legacyLabel of ["benign_keratosis", "dermatofibroma", "vascular_lesion"]) {
  if (labelSource.includes(legacyLabel)) {
    violations.push(`src/lib/labels.ts still exposes legacy label ${legacyLabel}`);
  }
}

for (const forbiddenFile of forbiddenFiles) {
  if (
    sourceFiles.some(
      (file) =>
        relative(fileURLToPath(mobileRoot), fileURLToPath(file)).replaceAll("\\", "/") ===
        forbiddenFile,
    )
  ) {
    violations.push(`${forbiddenFile} still exists`);
  }
}

if (violations.length) {
  throw new Error(`Privacy-first public boundary failed:\n${violations.join("\n")}`);
}

console.log("Privacy-first mobile boundary is stateless and account-free.");

async function collectSourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const url = new URL(`${entry.name}${entry.isDirectory() ? "/" : ""}`, directory);
    if (entry.isDirectory()) files.push(...(await collectSourceFiles(url)));
    else if (/\.(?:ts|tsx)$/.test(entry.name)) files.push(url);
  }
  return files;
}
