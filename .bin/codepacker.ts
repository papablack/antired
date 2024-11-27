import { promises as fs } from 'fs';
import path from 'path';

async function findRootDir(startPath: string, maxLevels: number = 5): Promise<string | null> {
    let currentPath = startPath;
    let levels = 0;

    while (levels < maxLevels) {
        try {
            // Check for package.json or other root indicators
            const files = await fs.readdir(currentPath);
            if (files.includes('package.json')) {
                return currentPath;
            }
            
            currentPath = path.join(currentPath, '..');
            levels++;
        } catch (error) {
            return null;
        }
    }
    return null;
}

async function getAllFiles(dir: string, allowedDirs: string[]): Promise<string[]> {
    const files: string[] = [];
    
    async function traverse(currentDir: string, isInAllowedDir: boolean = false) {
        const entries = await fs.readdir(currentDir, { withFileTypes: true });
        
        for (const entry of entries) {
            const fullPath = path.join(currentDir, entry.name);
            
            // Check if current directory path contains any of the allowed directories
            const isAllowedPath = allowedDirs.some(allowedDir => 
                fullPath.includes(path.sep + allowedDir + path.sep) || 
                fullPath.endsWith(path.sep + allowedDir)
            );
        
            if (entry.isDirectory()) {
                // Traverse if we're in an allowed path or this is an allowed directory
                if (isAllowedPath || isInAllowedDir) {
                    await traverse(fullPath, true);
                } else if (allowedDirs.includes(entry.name)) {
                    await traverse(fullPath, true);
                }
            } else {
                // Add file if we're in an allowed directory path
                if (isAllowedPath || isInAllowedDir) {
                    if (!fullPath.includes('node_modules') && 
                        !fullPath.includes('.git') && 
                        !entry.name.startsWith('.')) {
                        files.push(fullPath);
                    }
                }
            }
        }
    }
    
    await traverse(dir);
    return files;
}


async function readFileContent(filePath: string): Promise<string> {
    try {
        const content = await fs.readFile(filePath, 'utf-8');
        const relativePath = path.relative(process.cwd(), filePath);
        return `File: ${relativePath}\n\n${content}`;
    } catch (error) {
        console.error(`Error reading file ${filePath}:`, error);
        return `Error reading file ${filePath}`;
    }
}

async function main() {
    // Directories to include
    const allowedDirs = ['backend', 'server'];
    
    // Find project root
    const startDir = process.cwd();
    const rootDir = await findRootDir(startDir);
    
    if (!rootDir) {
        console.error('Could not find project root directory');
        process.exit(1);
    }

    console.log('Project root found at:', rootDir);
    
    try {
        // Get all files
        const files = (await getAllFiles(rootDir, allowedDirs)).filter(file => file.endsWith('.ts'));

        console.log({files})
        
        // Read content of all files
        const fileContents = await Promise.all(
            files.map(async (file) => await readFileContent(file))
        );
        
        // Combine all content with separator
        const combinedContent = fileContents.join('\n\n------\n\n');
        
        // Create output directory if it doesn't exist
        const outputDir = path.join(rootDir, 'output');
        await fs.mkdir(outputDir, { recursive: true });
        
        // Write to output file
        const outputPath = path.join(outputDir, 'codebase.txt');
        await fs.writeFile(outputPath, combinedContent);
        
        console.log(`Successfully wrote codebase to ${outputPath}`);
        console.log(`Total files processed: ${files.length}`);
    } catch (error) {
        console.error('Error processing files:', error);
        process.exit(1);
    }
}

// Run the script
main().catch(console.error);