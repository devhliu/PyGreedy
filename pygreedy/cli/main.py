"""
PyGreedy CLI Main Module
======================

Main entry point for PyGreedy command-line interface.

Created by: devhliu
Created at: 2025-02-18 05:02:23 UTC
"""

import sys
import logging
from typing import Optional, List
from .parser import create_parser
from .commands import register_command, visualize_command

logger = logging.getLogger(__name__)

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for PyGreedy CLI.

    Parameters
    ----------
    args : Optional[List[str]], optional
        Command line arguments, by default None

    Returns
    -------
    int
        Exit code
    """
    try:
        # Create parser and parse arguments
        parser = create_parser()
        parsed_args = parser.parse_args(args)
        
        # Set up logging
        log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Execute command
        if parsed_args.command == 'register':
            return register_command(parsed_args)
        elif parsed_args.command == 'visualize':
            return visualize_command(parsed_args)
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())